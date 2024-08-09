library(shiny)
library(ggplot2)  # Loading ggplot2 first to avoid masking issues
library(plotly)


# Kernel functions
linear_kernel <- function(x1, x2) {
  sum(x1 * x2)
}

polynomial_kernel <- function(x1, x2, degree) {
  (1 + sum(x1 * x2))^degree
}

rbf_kernel <- function(x1, x2, gamma) {
  exp(-gamma * sum((x1 - x2)^2))
}

sigmoid_kernel <- function(x1, x2, gamma, c) {
  tanh(gamma * sum(x1 * x2) + c)
}

# UI
ui <- fluidPage(
  titlePanel("Kernel Function Visualization"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("kernel_type", "Select Kernel", 
                  choices = c("Linear", "Polynomial", "RBF", "Sigmoid")),
      conditionalPanel(
        condition = "input.kernel_type == 'Polynomial'",
        sliderInput("degree", "Degree", min = 1, max = 10, value = 3)
      ),
      conditionalPanel(
        condition = "input.kernel_type == 'RBF' || input.kernel_type == 'Sigmoid'",
        sliderInput("gamma", "Gamma", min = 0.1, max = 2.0, value = 0.5, step = 0.1)
      ),
      conditionalPanel(
        condition = "input.kernel_type == 'Sigmoid'",
        sliderInput("c", "C", min = -2.0, max = 2.0, value = 1.0, step = 0.1)
      )
    ),
    
    mainPanel(
      plotlyOutput("kernel_plot")
    )
  )
)

# Server
server <- function(input, output) {
  output$kernel_plot <- renderPlotly({
    x <- seq(-5, 5, length.out = 50)
    y <- seq(-5, 5, length.out = 50)
    grid <- expand.grid(x = x, y = y)
    
    kernel_func <- switch(input$kernel_type,
                          "Linear" = function(x) linear_kernel(c(x[1], x[2]), c(1, 1)),
                          "Polynomial" = function(x) polynomial_kernel(c(x[1], x[2]), c(1, 1), input$degree),
                          "RBF" = function(x) rbf_kernel(c(x[1], x[2]), c(1, 1), input$gamma),
                          "Sigmoid" = function(x) sigmoid_kernel(c(x[1], x[2]), c(1, 1), input$gamma, input$c))
    
    z <- apply(grid, 1, kernel_func)
    
    plot_ly(x = x, y = y, z = matrix(z, nrow = length(x), ncol = length(y))) %>%
      add_surface() %>%
      layout(scene = list(xaxis = list(title = "X"),
                          yaxis = list(title = "Y"),
                          zaxis = list(title = "K(X, [1,1])")),
             title = paste(input$kernel_type, "Kernel"))
  })
}

# Run the app
shinyApp(ui = ui, server = server)
