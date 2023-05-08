import Foundation

/// Which kind of interpolation to use to interpolate 3h values to 1h values
enum Interpolation2StepType {
    // Simple linear interpolation
    case linear
    // Just copy the next value. Which means it is backwards filled. n+1 = n
    case nearest
    // Use solar radiation interpolation
    case solar_backwards_averaged
    // Use hemite interpolation
    case hermite(bounds: ClosedRange<Float>?)
    /// Hermite interpolation but for backward averaged data. Used for latent heat flux
    case hermite_backwards_averaged(bounds: ClosedRange<Float>?)
}

extension Array3DFastTime {
    mutating func interpolate2Steps(type: Interpolation2StepType, positions: [Int], grid: Gridable, locationRange: Range<Int>, run: Timestamp, dtSeconds: Int) {
        var d2 = Array2DFastTime(data: data, nLocations: nLocations*nLevel, nTime: nTime)
        d2.interpolate2Steps(type: type, positions: positions, grid: grid, locationRange: locationRange, run: run, dtSeconds: dtSeconds)
        data = d2.data
    }
    
    mutating func interpolate1Step(interpolation: ReaderInterpolation, interpolationHours: [Int], width: Int, time: TimerangeDt, grid: Gridable, locationRange: Range<Int>) {
        var d2 = Array2DFastTime(data: data, nLocations: nLocations*nLevel, nTime: nTime)
        d2.interpolate1Step(interpolation: interpolation, interpolationHours: interpolationHours, width: width, time: time, grid: grid, locationRange: locationRange)
        data = d2.data
    }
    
    mutating func deaccumulateOverTime(slidingWidth: Int, slidingOffset: Int) {
        var d2 = Array2DFastTime(data: data, nLocations: nLocations*nLevel, nTime: nTime)
        d2.deaccumulateOverTime(slidingWidth: slidingWidth, slidingOffset: slidingOffset)
        data = d2.data
    }
    
    mutating func deavergeOverTime(slidingWidth: Int, slidingOffset: Int) {
        var d2 = Array2DFastTime(data: data, nLocations: nLocations*nLevel, nTime: nTime)
        d2.deavergeOverTime(slidingWidth: slidingWidth, slidingOffset: slidingOffset)
        data = d2.data
    }
}


extension Array2DFastTime {
    /// Interpolate missing values for 1 hourly data that only has 3 hourly data at `positions`.
    mutating func interpolate2Steps(type: Interpolation2StepType, positions: [Int], grid: Gridable, locationRange: Range<Int>, run: Timestamp, dtSeconds: Int) {
        switch type {
        case .linear:
            interpolate2StepsLinear(positions: positions)
        case .nearest:
            interpolate2StepsNearest(positions: positions)
        case .solar_backwards_averaged:
            interpolate2StepsSolarBackwards(positions: positions, grid: grid, locationRange: locationRange, run: run, dtSeconds: dtSeconds)
        case .hermite(let bounds):
            interpolate2StepsHermite(positions: positions, bounds: bounds)
        case .hermite_backwards_averaged(let bounds):
            interpolate2StepsHermiteBackwardsAveraged(positions: positions, bounds: bounds)
        }
    }
    
    mutating func deavergeOverTime(slidingWidth: Int, slidingOffset: Int) {
        for l in 0..<nLocations {
            for start in stride(from: slidingOffset, to: nTime, by: slidingWidth) {
                var prev = self[l, start].isNaN ? 0 : self[l, start]
                var prevH = 1
                var skipped = 0
                for hour in start+1 ..< min(start+slidingWidth, nTime) {
                    let d = self[l, hour]
                    let h = hour-start+1
                    if d.isNaN {
                        skipped += 1
                        continue
                    }
                    self[l, hour] = (d * Float(h / (skipped+1)) - prev * Float(prevH / (skipped+1)))
                    prev = d
                    prevH = h
                    skipped = 0
                }
            }
        }
    }
    
    /// Note: Enforces >0
    mutating func deaccumulateOverTime(slidingWidth: Int, slidingOffset: Int) {
        for l in 0..<nLocations {
            for start in stride(from: slidingOffset, to: nTime, by: slidingWidth) {
                for hour in stride(from: min(start + slidingWidth, nTime) - 1, through: start + 1, by: -1) {
                    let current = self[l, hour]
                    let previous = self[l, hour-1]
                    if previous.isNaN, hour-2 >= 0 {
                        // allow 1x missing value
                        // This is a bit hacky, but the case is only present for a single timestep at the end of ARPEGE WORLD
                        let previous = self[l, hour-2]
                        self[l, hour] = previous.isNaN ? current : max(current - previous, 0) / 2
                        continue
                    }
                    // due to floating point precision, it can become negative
                    self[l, hour] = previous.isNaN ? current : max(current - previous, 0)
                }
            }
        }
    }
    
    /// 2 poisitions are interpolated in one step. Steps should align to `hour % 3 == 1`
    mutating func interpolate2StepsLinear(positions: [Int]) {
        for l in 0..<nLocations {
            for hour in positions {
                let prev = self[l, hour-1]
                let next = self[l, hour+2]
                self[l, hour] = prev * 2/3 + next * 1/3
                self[l, hour+1] = prev * 1/3 + next * 2/3
            }
        }
    }
    
    /// 2 poisitions are interpolated in one step. Steps should align to `hour % 3 == 1`
    mutating func interpolate2StepsHermite(positions: [Int], bounds: ClosedRange<Float>?) {
        for l in 0..<nLocations {
            for hour in positions {
                let A = self[l, hour-4 < 0 ? hour-1 : hour-4]
                let B = self[l, hour-1]
                let C = self[l, hour+2]
                let D = self[l, hour+4 >= nTime ? hour+2 : hour+5]
                let a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0
                let b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0
                let c = -A/2.0 + C/2.0
                let d = B
                let x0 = a*0.3*0.3*0.3 + b*0.3*0.3 + c*0.3 + d
                let x1 = a*0.6*0.6*0.6 + b*0.6*0.6 + c*0.6 + d
                if let bounds = bounds {
                    self[l, hour] = Swift.min(Swift.max(x0, bounds.lowerBound), bounds.upperBound)
                    self[l, hour+1] = Swift.min(Swift.max(x1, bounds.lowerBound), bounds.upperBound)
                } else {
                    self[l, hour] = x0
                    self[l, hour+1] = x1
                }
            }
        }
    }
    
    /// 2 poisitions are interpolated in one step. Steps should align to `hour % 3 == 1`
    mutating func interpolate2StepsHermiteBackwardsAveraged(positions: [Int], bounds: ClosedRange<Float>?) {
        /// basically shift the backwards averaged to the center and then do hermite
        for l in 0..<nLocations {
            for hour in positions {
                let A = self[l, hour-5 < 0 ? hour-2 : hour-5]
                let B = self[l, hour-2]
                let C = self[l, hour+2]
                let D = self[l, hour+4 >= nTime ? hour+2 : hour+5]
                let a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0
                let b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0
                let c = -A/2.0 + C/2.0
                let d = B
                let xm1 = a*0.3*0.3*0.3 + b*0.3*0.3 + c*0.3 + d
                let x0 = a*0.6*0.6*0.6 + b*0.6*0.6 + c*0.6 + d
                let x1 = C
                if let bounds = bounds {
                    self[l, hour-1] = Swift.min(Swift.max(xm1, bounds.lowerBound), bounds.upperBound)
                    self[l, hour] = Swift.min(Swift.max(x0, bounds.lowerBound), bounds.upperBound)
                    self[l, hour+1] = Swift.min(Swift.max(x1, bounds.lowerBound), bounds.upperBound)
                } else {
                    self[l, hour-1] = xm1
                    self[l, hour] = x0
                    self[l, hour+1] = C
                }
            }
        }
    }
    
    /// 2 poisitions are interpolated in one step. Steps should align to `hour % 3 == 1`
    mutating func interpolate2StepsNearest(positions: [Int]) {
        // fill with next hour. For weather code, we fill with the next hour, because this represents precipitation
        for l in 0..<nLocations {
            for hour in positions {
                let next = self[l, hour+2]
                self[l, hour] = next
                self[l, hour+1] = next
            }
        }
    }
    
    /// 2 poisitions are interpolated in one step. Steps should align to `hour % 3 == 1`
    mutating func interpolate2StepsSolarBackwards(positions: [Int], grid: Gridable, locationRange: Range<Int>, run: Timestamp, dtSeconds: Int) {
        // Solar backwards averages data. Data needs to be deaveraged before
        // First the clear sky index KT is calaculated (KT based on extraterrestrial radiation)
        // clearsky index is hermite interpolated and then back to actual radiation
        
        /// Which range of hours solar radiation data is required
        let solarHours = positions.minAndMax().map { $0.min - 4 ..< $0.max + 7 } ?? 0..<0
        let solarTime = TimerangeDt(start: run.add(solarHours.lowerBound * dtSeconds), nTime: solarHours.count, dtSeconds: dtSeconds)
        
        /// solar factor, backwards averaged over dt
        let solar2d = Zensun.calculateRadiationBackwardsAveraged(grid: grid, locationRange: locationRange, timerange: solarTime)
        
        /// Instead of caiculating solar radiation for the entire grid, itterate through a smaller grid portion
        for l in 0..<nLocations {
            for hour in positions {
                let sHour = hour - solarHours.lowerBound
                let sPos = l % locationRange.count
                // point C and D are still 3 h averages
                let solC1 = solar2d[sPos, sHour + 0]
                let solC2 = solar2d[sPos, sHour + 1]
                let solC3 = solar2d[sPos, sHour + 2]
                let solC = (solC1 + solC2 + solC3) / 3
                // At low radiaiton levels it is impossible to estimate KT indices
                let C = solC <= 0.005 ? 0 : min(self[l, hour+2] / solC, 1100)
                
                let solB = solar2d[sPos, sHour - 1]
                let B = solB <= 0.005 ? 0 : min(self[l, hour-1] / solB, 1100)
                
                let solA = solar2d[sPos, sHour - 4]
                let A = solA <= 0.005 ? 0 : hour-4 < 0 ? B : min((self[l, hour-4] / solA), 1100)
                
                let solD1 = solar2d[sPos, sHour + 3]
                let solD2 = solar2d[sPos, sHour + 4]
                let solD3 = solar2d[sPos, sHour + 5]
                let solD = (solD1 + solD2 + solD3) / 3
                let D = solD <= 0.005 ? 0 : hour+4 >= nTime ? C : min((self[l, hour+5] / solD), 1100)
                
                let a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0
                let b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0
                let c = -A/2.0 + C/2.0
                let d = B
                
                self[l, hour] = (a*0.3*0.3*0.3 + b*0.3*0.3 + c*0.3 + d) * solC1
                self[l, hour+1] = (a*0.6*0.6*0.6 + b*0.6*0.6 + c*0.6 + d) * solC2
                self[l, hour+2] = C * solC3
            }
        }
    }
}


extension Array2DFastTime {
    /// Used in ECMWF and MeteoFrance
    /// interpolate 1 missing step.. E.g. `DDDDDD-D-D-D-D-D`
    /// `dt` can be used to set element spacing E.g. `DxDxDxDxDxDx-xDx-xDx-xDx-xDx-xD` whith dt=1 all `x` positions will be ignored
    mutating func interpolate1Step(interpolation: ReaderInterpolation, interpolationHours: [Int], width: Int, time: TimerangeDt, grid: Gridable, locationRange: Range<Int>) {
        switch interpolation {
        case .nearest:
            // take previous hour
            for l in 0..<nLocations {
                for hour in interpolationHours {
                    let prev = self[l, hour-1*width]
                    self[l, hour] = prev
                }
            }
        case .backwards_sum:
            // do not divide precip amount, because data is still averaged at this point
            fallthrough
        case .linear:
            for l in 0..<nLocations {
                for hour in interpolationHours {
                    let prev = self[l, hour-1*width]
                    let next = self[l, hour+1*width]
                    self[l, hour] = prev * 1/2 + next * 1/2
                }
            }
        case .hermite:
            for l in 0..<nLocations {
                for hour in interpolationHours {
                    let A = self[l, hour-3*width < 0 ? hour-1*width : hour-3*width]
                    let B = self[l, hour-1*width]
                    let C = self[l, hour+1*width]
                    let D = self[l, hour+2*width >= nTime ? hour+1*width : hour+3*width]
                    let a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0
                    let b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0
                    let c = -A/2.0 + C/2.0
                    let d = B
                    self[l, hour] = a*0.5*0.5*0.5 + b*0.5*0.5 + c*0.5 + d
                }
            }
        case .solar_backwards_averaged:
            // Solar backwards averages data. Data needs to be deaveraged before
            // First the clear sky index KT is calaculated (KT based on extraterrestrial radiation)
            // clearsky index is hermite interpolated and then back to actual radiation
            
            if interpolationHours.isEmpty {
                return
            }
            
            /// Which range of hours solar radiation data is required
            let solarHours = interpolationHours.minAndMax().map { $0.min - 3 * width ..< $0.max + 5 * width } ?? 0..<0
            let solarTime = TimerangeDt(
                start: time.range.lowerBound.add(solarHours.lowerBound * time.dtSeconds),
                nTime: solarHours.count / width,
                dtSeconds: time.dtSeconds * width
            )
            
            /// solar factor, backwards averaged over dt
            let solar2d = Zensun.calculateRadiationBackwardsAveraged(grid: grid, locationRange: locationRange, timerange: solarTime)
            
            /// Instead of caiculating solar radiation for the entire grid, itterate through a smaller grid portion
            for l in 0..<nLocations {
                for hour in interpolationHours {
                    let sHour = (hour - solarHours.lowerBound) / width
                    let sPos = l % locationRange.count
                    // point C and D are still 2 step averages
                    let solC1 = solar2d[sPos, sHour + 0]
                    let solC2 = solar2d[sPos, sHour + 1]
                    let solC = (solC1 + solC2) / 2
                    // At low radiation levels it is impossible to estimate KT indices
                    var C = solC <= 0.005 ? 0 : min(self[l, hour+1*width] / solC, 1100)
                    
                    let solB = solar2d[sPos, sHour - 1]
                    var B = solB <= 0.005 ? C : min(self[l, hour-1*width] / solB, 1100)
                    
                    if C == 0 && B > 0 {
                        C = B
                    }
                    
                    let solA = solar2d[sPos, sHour - 3]
                    var A = solA <= 0.005 ? B : hour-3 < 0 ? B : min((self[l, hour-3*width] / solA), 1100)
                    
                    if C == 0 && A > 0 {
                        B = A
                        C = A
                    }
                    
                    let solD1 = solar2d[sPos, sHour + 2]
                    let solD2 = solar2d[sPos, sHour + 3]
                    let solD = (solD1 + solD2) / 2
                    let D = solD <= 0.005 ? C : hour+3 > nTime ? C : min((self[l, hour+3*width] / solD), 1100)
                    
                    // Espcially for 6h values, aggressively try to find any KT index that works
                    // As a future improvement, the clearsky radiation could be approximated by cloud cover total as an additional input
                    // This could improve morning/evening kt approximations
                    if C == 0 && D > 0 {
                        A = D
                        B = D
                        C = D
                    }
                    
                    let a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0
                    let b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0
                    let c = -A/2.0 + C/2.0
                    let d = B
                    
                    self[l, hour] = (a*0.5*0.5*0.5 + b*0.5*0.5 + c*0.5 + d) * solC1
                    self[l, hour+1*width] = C * solC2
                }
            }
        }
    }
}
