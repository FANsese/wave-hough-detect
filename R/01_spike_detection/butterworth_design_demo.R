require(signal)
bf <- butter(3, 0.3, type="high")                          # 10 Hz low-pass filter
t <- seq(0, 1, len = 100)                     # 1 second sample
# x <- sin(2*pi*t*2.3) + 0.25*rnorm(length(t))  # 2.3 Hz sinusoid+noise
# x <- sin(2*pi*t*10) + 0.1*(1:length(t))  # 2.3 Hz sinusoid+noise
x <- sin(2*pi*t*20) + sin(2*pi*t*2)  # 2.3 Hz sinusoid+noise
x2 <- sin(2*pi*t*10)
z <- filter(bf, x) # apply filter
plot(t, x, type = "l")
lines(t, z, col = "red")
# lines(t, x2, col = "green")
legend("bottomleft", c("original", "filtered"), col=c("black", "red"), lty=c(1,1))
# dev.copy(pdf,'butter_example2.pdf') 
# dev.off()
