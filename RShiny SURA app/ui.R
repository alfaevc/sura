library(datasets)


shinyUI(fluidPage(
  titlePanel("PUBMED Trial/Non-Trial Predictor"),
  sidebarLayout(
    sidebarPanel(

      selectInput("inputType", "Please Select Input Type",
                  choices = c("Text File", "Title", "Abstract")),
      conditionalPanel(condition = "input.inputType == 'Title'",
                       textInput("Title", "Please Enter The Title", "")),
      conditionalPanel(condition = "input.inputType == 'Abstract'",
                       textInput("Abstract", "Please Enter The Abstract", "")),
      conditionalPanel(condition = "input.inputType == 'Text File'",
                       fileInput("file1", "Choose Text File",
                                 accept = c(
                                   "text/csv",
                                   "text/comma-separated-values,text/plain",
                                   ".csv")),
      ),
      tags$hr(),
      radioButtons("view", "Output View",
                   c("Simple" = "S", "Detailed" = "D")),
      actionButton("predictButton", "Predict")
      
      #checkboxInput("detail", "detailedView", TRUE)
     ),
    mainPanel(
      tableOutput("contents")
    )
  )
)
)