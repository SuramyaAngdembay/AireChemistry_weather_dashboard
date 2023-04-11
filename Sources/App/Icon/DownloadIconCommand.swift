import Foundation
import Vapor
import SwiftNetCDF
import SwiftPFor2D
import Dispatch
import CHelper


struct CdoHelper {
    let cdo: CdoIconGlobal?
    let grid: Gridable
    let domain: IconDomains
    
    init(domain: IconDomains, logger: Logger, client: HTTPClient) async throws {
        // icon global needs resampling to plate carree
        cdo = domain == .icon ? try await CdoIconGlobal(logger: logger, workDirectory: domain.downloadDirectory, client: client) : nil
        grid = domain.grid
        self.domain = domain
    }
    
    func readGrib2(_ filename: String) throws -> [Float] {
        let tempNc = "\(filename).nc"
        
        if let cdo = cdo {
            // resample to regular latlon (icon global)
            try cdo.remap(in: filename, out: tempNc)
        } else {
            // just convert grib2 to netcdf
            try Process.grib2ToNetcdf(in: filename, out: tempNc)
        }
        let data = try readNetCdf(path: tempNc)
        try FileManager.default.removeItem(atPath: tempNc)
        return data
    }
    
    func readNetCdf(path: String) throws -> [Float] {
        guard let nc = try NetCDF.open(path: path, allowUpdate: false) else {
            fatalError("File \(path) does not exist")
        }
        guard let v = nc.getVariables().first(where: {$0.dimensions.count >= 3}) else {
            fatalError("Could not find data variable with 3d/4d data")
        }
        precondition(v.dimensions[v.dimensions.count-1].length == grid.nx)
        precondition(v.dimensions[v.dimensions.count-2].length == grid.ny)
        guard let varFloat = v.asType(Float.self) else {
            fatalError("Netcdf variable is not float type")
        }
        /// icon-d2 total precip, aswdir and aswdifd have 15 minutes values
        let offset = v.dimensions.count == 3 ? [0,0,0] : [0,0,0,0]
        let count = v.dimensions.count == 3 ? [1,grid.ny,grid.nx] : [1,1,grid.ny,grid.nx]
        var d = try varFloat.read(offset: offset, count: count)
        for x in d.indices {
            if d[x] < -100000000 {
                d[x] = .nan
            }
        }
        return d
    }
}

struct DownloadIconCommand: AsyncCommandFix {
    enum VariableGroup: String, RawRepresentable, CaseIterable {
        case all
        case surface
        case modelLevel
        case pressureLevel
        case pressureLevelGt500
        case pressureLevelLtE500
    }
    
    struct Signature: CommandSignature {
        @Argument(name: "domain")
        var domain: String

        @Option(name: "run")
        var run: String?

        @Flag(name: "skip-existing")
        var skipExisting: Bool
        
        @Option(name: "group")
        var group: String?
        
        @Option(name: "only-variables")
        var onlyVariables: String?
    }

    var help: String {
        "Download a specified icon model run"
    }
    
    /**
     Convert surface elevation. Out of grid positions are NaN. Sea grid points are -999.
     */
    func convertSurfaceElevation(application: Application, domain: IconDomains, run: Timestamp) async throws {
        let logger = application.logger
        if FileManager.default.fileExists(atPath: domain.surfaceElevationFileOm) {
            return
        }
        
        let gridType = domain == .icon ? "icosahedral" : "regular-lat-lon"
        let downloadDirectory = domain.downloadDirectory
        try FileManager.default.createDirectory(atPath: downloadDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: domain.omfileDirectory, withIntermediateDirectories: true)
        
        let domainPrefix = "\(domain.rawValue)_\(domain.region)"
        let cdo = try await CdoHelper(domain: domain, logger: logger, client: application.dedicatedHttpClient)
        
        // https://opendata.dwd.de/weather/nwp/icon/grib/00/t_2m/icon_global_icosahedral_single-level_2022070800_000_T_2M.grib2.bz2
        // https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/icon-eu_europe_regular-lat-lon_single-level_2022072000_000_T_2M.grib2.bz2
        let serverPrefix = "http://opendata.dwd.de/weather/nwp/\(domain.rawValue)/grib/\(run.hour.zeroPadded(len: 2))/"
        let dateStr = run.format_YYYYMMddHH
        let curl = Curl(logger: logger, client: application.dedicatedHttpClient, deadLineHours: domain == .iconD2 ? 2 : 5)
        // surface elevation
        // https://opendata.dwd.de/weather/nwp/icon/grib/00/hsurf/icon_global_icosahedral_time-invariant_2022072400_HSURF.grib2.bz2
        if !FileManager.default.fileExists(atPath: domain.surfaceElevationFileOm) {
            let file: String
            if domain == .iconD2 {
                file = "\(serverPrefix)hsurf/\(domainPrefix)_\(gridType)_time-invariant_\(dateStr)_000_0_hsurf.grib2.bz2"
            } else {
                file = "\(serverPrefix)hsurf/\(domainPrefix)_\(gridType)_time-invariant_\(dateStr)_HSURF.grib2.bz2"
            }
            try await curl.download(
                url: file,
                toFile: "\(downloadDirectory)time-invariant_HSURF.grib2",
                bzip2Decode: true
            )
        
            // land fraction
            let file2: String
            if domain == .iconD2 {
                file2 = "\(serverPrefix)fr_land/\(domainPrefix)_\(gridType)_time-invariant_\(dateStr)_000_0_fr_land.grib2.bz2"
            } else {
                file2 = "\(serverPrefix)fr_land/\(domainPrefix)_\(gridType)_time-invariant_\(dateStr)_FR_LAND.grib2.bz2"
            }
            try await curl.download(
                url: file2,
                toFile: "\(downloadDirectory)time-invariant_FR_LAND.grib2",
                bzip2Decode: true
            )
        }
        
        // use special numbers for SEA grid points?
        var hsurf = try cdo.readGrib2("\(domain.downloadDirectory)time-invariant_HSURF.grib2")
        let landFraction = try cdo.readGrib2("\(domain.downloadDirectory)time-invariant_FR_LAND.grib2")
        
        // Set all sea grid points to -999
        precondition(hsurf.count == landFraction.count)
        for i in hsurf.indices {
            if landFraction[i] < 0.5 {
                hsurf[i] = -999
            }
        }
        
        try OmFileWriter(dim0: domain.grid.ny, dim1: domain.grid.nx, chunk0: 20, chunk1: 20).write(file: domain.surfaceElevationFileOm, compressionType: .p4nzdec256, scalefactor: 1, all: hsurf)
    }
    
    
    /// Download ICON global, eu and d2 *.grid2.bz2 files
    func downloadIcon(application: Application, domain: IconDomains, run: Timestamp, skipFilesIfExisting: Bool, variables: [IconVariableDownloadable]) async throws {
        let logger = application.logger
        let gridType = domain == .icon ? "icosahedral" : "regular-lat-lon"
        let downloadDirectory = domain.downloadDirectory
        try FileManager.default.createDirectory(atPath: downloadDirectory, withIntermediateDirectories: true)
        
        let domainPrefix = "\(domain.rawValue)_\(domain.region)"
        let cdo = try await CdoHelper(domain: domain, logger: logger, client: application.dedicatedHttpClient)
        
        // https://opendata.dwd.de/weather/nwp/icon/grib/00/t_2m/icon_global_icosahedral_single-level_2022070800_000_T_2M.grib2.bz2
        // https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/icon-eu_europe_regular-lat-lon_single-level_2022072000_000_T_2M.grib2.bz2
        let serverPrefix = "http://opendata.dwd.de/weather/nwp/\(domain.rawValue)/grib/\(run.hour.zeroPadded(len: 2))/"
        let dateStr = run.format_YYYYMMddHH
        let curl = Curl(logger: logger, client: application.dedicatedHttpClient, deadLineHours: domain == .iconD2 ? 2 : 5, waitAfterLastModified: 120)
        let nLocationsPerChunk = OmFileSplitter(basePath: domain.omfileDirectory, nLocations: domain.grid.count, nTimePerFile: domain.omFileLength, yearlyArchivePath: nil).nLocationsPerChunk
        
        let writer = OmFileWriter(dim0: 1, dim1: domain.grid.count, chunk0: 1, chunk1: nLocationsPerChunk)
        
        var grib2d = GribArray2D(nx: domain.grid.nx, ny: domain.grid.ny)

        let forecastSteps = domain.getDownloadForecastSteps(run: run.hour)
        for hour in forecastSteps {
            logger.info("Downloading hour \(hour)")
            let h3 = hour.zeroPadded(len: 3)
            for variable in variables {
                if hour == 0 && variable.skipHour0(domain: domain, forDownload: true) {
                    continue
                }
                guard let v = variable.getVarAndLevel(domain: domain) else {
                    continue
                }
                let level = v.level.map({"_\($0)"}) ?? ""
                let leveld2 = v.level.map({"_\($0)"}) ?? "_2d"
                let filenameFrom = domain != .iconD2 ?
                    "\(domainPrefix)_\(gridType)_\(v.cat)_\(dateStr)_\(h3)\(level)_\(v.variable.uppercased()).grib2.bz2" :
                    "\(domainPrefix)_\(gridType)_\(v.cat)_\(dateStr)_\(h3)\(leveld2)_\(v.variable).grib2.bz2"
                
                let url = "\(serverPrefix)\(v.variable)/\(filenameFrom)"
                
                let filenameDest = "single-level_\(h3)_\(variable.omFileName.uppercased()).fpg"
                if skipFilesIfExisting && FileManager.default.fileExists(atPath: "\(downloadDirectory)\(filenameDest)") {
                    continue
                }
                
                var data: [Float]
                if domain == .icon {
                    // regrid from icosahedral to regular lat-lon
                    let gribFile = "\(downloadDirectory)\(variable.omFileName).grib2"
                    try await curl.download(
                        url: url,
                        toFile: gribFile,
                        bzip2Decode: true
                    )
                    // Uncompress bz2, reproject to regular grid, convert to netcdf and read into memory
                    // Especially reprojecting is quite slow, therefore we can better utilise the download time waiting for the next file
                    data = try cdo.readGrib2(gribFile)
                    try FileManager.default.removeItem(atPath: gribFile)
                } else {
                    // Use async in-memory download and decoding -> 4 times faster, but cannot regrid icosahedral data
                    let messages = try await curl.downloadGrib(url: url, bzip2Decode: true)
                    if domain == .iconD2 && messages.count > 1 {
                        // Write 15min D2 icon data
                        let downloadDirectory = IconDomains.iconD2_15min.downloadDirectory
                        try FileManager.default.createDirectory(atPath: downloadDirectory, withIntermediateDirectories: true)
                        for (i, message) in messages.enumerated() {
                            let h3 = (hour*4+i).zeroPadded(len: 3)
                            let filenameDest = "single-level_\(h3)_\(variable.omFileName.uppercased()).fpg"
                            try grib2d.load(message: message)
                            data = grib2d.array.data
                            try FileManager.default.removeItemIfExists(at: "\(downloadDirectory)\(filenameDest)")
                            if let fma = variable.multiplyAdd {
                                data.multiplyAdd(multiply: fma.multiply, add: fma.add)
                            }
                            let compression = variable.isAveragedOverForecastTime || variable.isAccumulatedSinceModelStart ? CompressionType.fpxdec32 : .p4nzdec256
                            try writer.write(file: "\(downloadDirectory)\(filenameDest)", compressionType: compression, scalefactor: variable.scalefactor, all: data)
                        }
                    }
                    let message = messages[0]
                    try grib2d.load(message: message)
                    data = grib2d.array.data
                }
                
                // Write data as encoded floats to disk
                try FileManager.default.removeItemIfExists(at: "\(downloadDirectory)\(filenameDest)")
                
                // Scaling before compression with scalefactor
                if let fma = variable.multiplyAdd {
                    data.multiplyAdd(multiply: fma.multiply, add: fma.add)
                }
                
                //logger.info("Compressing and writing data to \(filenameDest)")
                let compression = variable.isAveragedOverForecastTime || variable.isAccumulatedSinceModelStart ? CompressionType.fpxdec32 : .p4nzdec256
                try writer.write(file: "\(downloadDirectory)\(filenameDest)", compressionType: compression, scalefactor: variable.scalefactor, all: data)
                
                // icon global downloads tend to use a lot of memory due to numerous allocations
                chelper_malloc_trim()
            }
        }
        curl.printStatistics()
    }

    /// unompress and remap
    /// Process variable after variable
    func convertIcon(logger: Logger, domain: IconDomains, run: Timestamp, variables: [IconVariableDownloadable]) throws {
        let downloadDirectory = domain.downloadDirectory
        let grid = domain.grid
        
        let forecastSteps = domain.getDownloadForecastSteps(run: run.hour)
        let nTime = domain.nForecastHours(run: run.hour)
        let nLocations = grid.count
        
        try FileManager.default.createDirectory(atPath: domain.omfileDirectory, withIntermediateDirectories: true)
        let om = OmFileSplitter(basePath: domain.omfileDirectory, nLocations: nLocations, nTimePerFile: domain.omFileLength, yearlyArchivePath: nil)
        let nLocationsPerChunk = om.nLocationsPerChunk
        //print("nLocationsPerChunk \(nLocationsPerChunk)... \(nLocations/nLocationsPerChunk) iterations")
        let ringtime = run.timeIntervalSince1970 / domain.dtSeconds ..< run.timeIntervalSince1970 / domain.dtSeconds + nTime

        // ICON global + eu only have 3h data after 78 hours
        // ICON global 6z and 18z have 120 instead of 180 forecast hours
        // Stategy: Read each variable in a spatial array and interpolate missing values
        // Afterwards merge into temporal data files
        
        var data2d = Array2DFastTime(nLocations: nLocationsPerChunk, nTime: nTime)
        var readTemp = [Float](repeating: .nan, count: nLocationsPerChunk)

        for variable in variables {
            guard variable.getVarAndLevel(domain: domain) != nil else {
                continue
            }
            let v = variable.omFileName.uppercased()
            let skip = variable.skipHour0(domain: domain, forDownload: false) ? 1 : 0
            let progress = ProgressTracker(logger: logger, total: nLocations, label: "Convert \(variable.rawValue)")
            
            let readers: [(hour: Int, reader: OmFileReader<MmapFile>)] = try forecastSteps.compactMap({ hour in
                if hour < skip {
                    return nil
                }
                let reader = try OmFileReader(file: "\(downloadDirectory)single-level_\(hour.zeroPadded(len: 3))_\(v).fpg")
                try reader.willNeed()
                return (hour, reader)
            })
            
            try om.updateFromTimeOrientedStreaming(variable: variable.omFileName, ringtime: ringtime, skipFirst: skip, smooth: 0, skipLast: 0, scalefactor: variable.scalefactor) { d0offset in
                
                let locationRange = d0offset ..< min(d0offset+nLocationsPerChunk, nLocations)
                data2d.data.fillWithNaNs()
                for reader in readers {
                    try reader.reader.read(into: &readTemp, arrayDim1Range: 0..<locationRange.count, arrayDim1Length: locationRange.count, dim0Slow: 0..<1, dim1: locationRange)
                    data2d[0..<data2d.nLocations, reader.hour /*/ domain.dtHours*/] = readTemp
                }
                
                // Deaverage radiation. Not really correct for 3h data after 81 hours, but interpolation will correct in the next step.
                if variable.isAveragedOverForecastTime {
                    data2d.deavergeOverTime(slidingWidth: data2d.nTime, slidingOffset: skip)
                }
                
                // interpolate missing timesteps. We always fill 2 timesteps at once
                // data looks like: DDDDDDDDDD--D--D--D--D--D
                let forecastStepsToInterpolate = (0..<nTime).compactMap { hour -> Int? in
                    if forecastSteps.contains(hour) || hour % 3 != 1 {
                        // process 2 timesteps at once
                        return nil
                    }
                    return hour
                }
                
                // Fill in missing hourly values after switching to 3h
                data2d.interpolate2Steps(type: variable.interpolationType, positions: forecastStepsToInterpolate, grid: domain.grid, locationRange: locationRange, run: run, dtSeconds: domain.dtSeconds)
                
                // De-accumulate precipitation
                if variable.isAccumulatedSinceModelStart {
                    data2d.deaccumulateOverTime(slidingWidth: data2d.nTime, slidingOffset: skip)
                }
                
                progress.add(locationRange.count)
                return data2d.data[0..<locationRange.count * nTime]
            }
            progress.finish()
        }
        logger.info("write init.txt")
        try "\(run.timeIntervalSince1970)".write(toFile: domain.initFileNameOm, atomically: true, encoding: .utf8)
    }

    func run(using context: CommandContext, signature: Signature) async throws {
        let start = DispatchTime.now()
        let domain = try IconDomains.load(rawValue: signature.domain)
        
        let run = try signature.run.flatMap(Timestamp.fromRunHourOrYYYYMMDD) ?? domain.lastRun
        
        let group = try VariableGroup.load(rawValueOptional: signature.group) ?? .all
        
        let onlyVariables: [IconVariableDownloadable]? = try signature.onlyVariables.map {
            try $0.split(separator: ",").map {
                if let variable = IconPressureVariable(rawValue: String($0)) {
                    return variable
                }
                return try IconSurfaceVariable.load(rawValue: String($0))
            }
        }
        
        /// 3 different variables sets to optimise download time:
        /// - surface variables with soil
        /// - model-level e.g. for 180m wind, they have a much larger dalay and sometimes are aborted
        /// - pressure level which take forever to download because it is too much data
        var groupVariables: [IconVariableDownloadable]
        switch group {
        case .all:
            groupVariables = IconSurfaceVariable.allCases + domain.levels.reversed().flatMap { level in
                IconPressureVariableType.allCases.map { variable in
                    IconPressureVariable(variable: variable, level: level)
                }
            }
        case .surface:
            groupVariables = IconSurfaceVariable.allCases.filter {
                !($0.getVarAndLevel(domain: domain)?.cat == "model-level")
            }
        case .modelLevel:
            groupVariables = IconSurfaceVariable.allCases.filter {
                $0.getVarAndLevel(domain: domain)?.cat == "model-level"
            }
        case .pressureLevel:
            groupVariables = domain.levels.reversed().flatMap { level in
                IconPressureVariableType.allCases.map { variable in
                    IconPressureVariable(variable: variable, level: level)
                }
            }
        case .pressureLevelGt500:
            groupVariables = domain.levels.reversed().flatMap { level in
                return level > 500 ? IconPressureVariableType.allCases.map { variable in
                    IconPressureVariable(variable: variable, level: level)
                } : []
            }
        case .pressureLevelLtE500:
            groupVariables = domain.levels.reversed().flatMap { level in
                return level <= 500 ? IconPressureVariableType.allCases.map { variable in
                    IconPressureVariable(variable: variable, level: level)
                } : []
            }
        }
        
        let variables = onlyVariables ?? groupVariables
                
        let logger = context.application.logger
        logger.info("Downloading domain '\(domain.rawValue)' run '\(run.iso8601_YYYY_MM_dd_HH_mm)'")
        try await convertSurfaceElevation(application: context.application, domain: domain, run: run)
        
        try await downloadIcon(application: context.application, domain: domain, run: run, skipFilesIfExisting: signature.skipExisting, variables: variables)
        try convertIcon(logger: logger, domain: domain, run: run, variables: variables)
        if domain == .iconD2 {
            // ICON-D2 download 15min data as well
            try convertIcon(logger: logger, domain: .iconD2_15min, run: run, variables: variables)
        }
        
        logger.info("Finished in \(start.timeElapsedPretty())")
    }
}

extension IconDomains {
    /// Based on the current time , guess the current run that should be available soon on the open-data server
    fileprivate var lastRun: Timestamp {
        let t = Timestamp.now()
        switch self {
        case .icon:
            // Icon has a delay of 2-3 hours after initialisation  with 4 runs a day
            return t.with(hour: ((t.hour - 2 + 24) % 24) / 6 * 6)
        case .iconEu:
            // Icon-eu has a delay of 2:40 hours after initialisation with 8 runs a day
            return t.with(hour: ((t.hour - 2 + 24) % 24) / 3 * 3)
        case .iconD2:
            // Icon d2 has a delay of 44 minutes and runs every 3 hours
            return t.with(hour: t.hour / 3 * 3)
        case .iconD2_15min:
            fatalError("ICON-D2 15minute data can not be downloaded individually")
        }
    }
}


/// Workaround to use async in commans
/// Wait for https://github.com/vapor/vapor/pull/2870
protocol AsyncCommandFix: Command {
    func run(using context: CommandContext, signature: Signature) async throws
}

extension AsyncCommandFix {
    func run(using context: CommandContext, signature: Signature) throws {
        // use same thread as downloader, do not use main loop
        /*let eventloop = context.application.dedicatedHttpClient.eventLoopGroup.next()
        let result = eventloop.flatSubmit {
            let promise = eventloop.makePromise(of: Void.self)
            promise.completeWithTask {
                try await run(using: context, signature: signature)
            }
            return promise.futureResult
        }
        try result.wait()*/
        
        // mainloop or dedicated loop make no difference apparently
        let eventloop = context.application.eventLoopGroup.next()
        let promise = eventloop.makePromise(of: Void.self)
        promise.completeWithTask {
            try await run(using: context, signature: signature)
        }
        try promise.futureResult.wait()
    }
}
