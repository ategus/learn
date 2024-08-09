# example.R
library(ggplot2)
# If plotly is loaded and causing issues, you can detach it
if("package:plotly" %in% search()) {
    detach("package:plotly", unload=TRUE)
}
print(ggplot(mpg, aes(x = displ, y = hwy, color = class)) + geom_point())

