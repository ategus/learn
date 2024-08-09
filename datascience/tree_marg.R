# Install and load required packages

library(igraph)
library(ggplot2)
library(ggraph)

# Node class
Node <- setRefClass("Node",
  fields = list(
    name = "character",
    probability = "matrix",
    children = "list",
    parent = "ANY",
    message_to_parent = "ANY"
  ),
  methods = list(
    initialize = function(name, probability) {
      name <<- name
      probability <<- probability
      children <<- list()
      parent <<- NULL
      message_to_parent <<- NULL
    }
  )
)

# Create tree function
create_tree <- function() {
  root <- Node$new("A", matrix(c(0.6, 0.4), nrow = 1))
  b <- Node$new("B", matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE))
  c <- Node$new("C", matrix(c(0.5, 0.5, 0.1, 0.9), nrow = 2, byrow = TRUE))
  d <- Node$new("D", matrix(c(0.8, 0.2, 0.3, 0.7), nrow = 2, byrow = TRUE))

  root$children <- list(b, c)
  b$parent <- root
  c$parent <- root
  b$children <- list(d)
  d$parent <- b

  return(root)
}

# Marginalize function
marginalize <- function(node) {
  if (length(node$children) == 0) {
    node$message_to_parent <- node$probability
    return(node$probability)
  }

  child_messages <- lapply(node$children, marginalize)
  child_messages <- do.call(cbind, child_messages)

  if (!is.null(node$parent)) {
    node$message_to_parent <- node$probability %*% apply(child_messages, 1, prod)
    return(node$message_to_parent)
  } else {
    return(node$probability * apply(child_messages, 1, prod))
  }
}

# Visualize tree function
visualize_tree <- function(root) {
  edges <- data.frame()
  nodes <- data.frame()

  traverse <- function(node) {
    nodes <<- rbind(nodes, data.frame(name = node$name, label = paste(node$name, "\n", paste(round(node$message_to_parent, 3), collapse = ", "))))
    for (child in node$children) {
      edges <<- rbind(edges, data.frame(from = node$name, to = child$name))
      traverse(child)
    }
  }

  traverse(root)

  g <- graph_from_data_frame(edges, vertices = nodes)

  ggraph(g, layout = "tree") +
    geom_edge_link(aes(label = paste("P(", substr(.N()$name, 1, 1), "|", substr(.N()$name, 2, 2), ")")),
