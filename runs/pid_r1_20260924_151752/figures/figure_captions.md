# Figure captions

## Error Time (`01_error_time.png`)

Lateral deviation of the detected lane centre from the image centre over the duration of the run. The shaded band is the control deadzone, inside which no steering correction is issued.

## Error Distribution (`02_error_distribution.png`)

Probability density of the lateral error, with the median marked. The distribution is reported by median and interquartile range rather than by a normal fit, because a vehicle oscillating about the lane centre produces a bimodal rather than a normal distribution.

## Error Cdf (`03_error_cdf.png`)

Empirical cumulative distribution of the absolute lateral error. The 95th percentile is a stricter accuracy statement than the mean, since it bounds all but the worst 5 % of frames.

## Pid Terms (`04_pid_terms.png`)

Individual contributions of the proportional, integral and derivative terms to the steering command. The proportional term dominates the transient response, the integral term removes the steady-state offset in sustained curves, and the derivative term opposes rapid changes and so damps overshoot.

## Controller Characteristic (`05_controller_characteristic.png`)

Steering command against measured lateral error for every frame. The bang-bang controller occupies three discrete levels; the PID controller produces a continuous, approximately proportional response, shown by the binned mean. Discrete levels are drawn with slight vertical jitter so that overlapping points remain visible.

## Frame Rate (`06_frame_rate.png`)

Instantaneous frame rate over the run (thin line) with a 15-frame moving average (thick line), and the corresponding distribution. The dashed line marks the 15 FPS real-time target of hypothesis H4.

## Latency (`07_latency.png`)

Distribution of the measured end-to-end latency from image capture to the issued motor command. The two reference lines are the thresholds stated in hypothesis H2.

## Stage Timing (`08_stage_timing.png`)

Mean execution time of each pipeline stage on the target hardware, with standard deviation. The telemetry display is excluded from the bars and reported separately, since it runs only during development and dominates the budget by an order of magnitude.

## Detection (`09_detection.png`)

Per-frame detection confidence with a 25-frame moving average. Tick marks along the bottom axis mark frames in which no lane was detected; the overall detection rate is stated inset.

## Motor Duty (`10_motor_duty.png`)

Commanded PWM duty cycle of the left and right motor channels. The shaded area is the differential between the two channels, which is what produces the yaw rate of the vehicle.

## Lane Positions (`11_lane_positions.png`)

Detected left and right lane boundary positions and the resulting lane centre, in image coordinates. Gaps in the boundary traces are frames where only one boundary was visible and the other was inferred from the estimated lane width.

## Error Spectrum (`12_error_spectrum.png`)

Amplitude spectrum of the lateral error, averaged over overlapping Hanning-windowed segments to suppress the bin-to-bin variance of a single periodogram. The marked peak is the dominant oscillation frequency: a lower peak amplitude corresponds to smoother steering, which is the quantitative form of hypothesis H1.

## Phase Portrait (`13_phase_portrait.png`)

Phase portrait of the lateral error against its rate of change, one panel per configuration, shaded by the density of visited states. The error is smoothed before differentiation because differentiating a pixel-quantised signal directly yields mostly noise. Trajectories concentrated near the origin indicate a well-damped controller; a closed ring indicates a sustained limit cycle.

## Latency Series (`14_latency_series.png`)

End-to-end latency per frame (thin) and its 25-frame moving average (thick). Periodic excursions typically indicate CPU contention or thermal throttling of the Raspberry Pi rather than algorithmic cost.
