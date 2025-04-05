# main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from animalposetracker import AnimalPoseTracker  # Import the generated UI class

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI components
        self.ui = AnimalPoseTracker()
        self.ui.setupUi(self)  # Set up the UI from designer file
        
        # Connect button signals to methods
        self.connect_actions()
    
    def connect_actions(self):
        """Connect UI buttons to their respective handler functions"""
        # Main project selection buttons
        self.ui.CreateNewProjectButton.clicked.connect(self.create_new_project)
        self.ui.LoadProjectButton.clicked.connect(self.load_project)
        self.ui.PublicDatasetsProjectButton.clicked.connect(self.open_public_datasets)
        
        # Menu bar actions
        self.ui.actionFileCreateNewProject.triggered.connect(self.create_new_project)
        self.ui.actionFileLoadProject.triggered.connect(self.load_project)
        self.ui.actionPublicDatasetsProject.triggered.connect(self.open_public_datasets)
        self.ui.actionExit.triggered.connect(self.close)  # Exit application
    
    # Handler methods for project selection
    def create_new_project(self):
        """Handle 'Create New Project' button click"""
        print("New project creation logic goes here")
        # TODO: Implement project creation workflow
    
    def load_project(self):
        """Handle 'Load Project' button click"""
        print("Project loading logic goes here")
        # TODO: Implement project loading dialog
    
    def open_public_datasets(self):
        """Handle 'Public Datasets' button click"""
        print("Public datasets access logic goes here")
        # TODO: Implement datasets browser

if __name__ == "__main__":
    # Initialize Qt application
    app = QApplication(sys.argv)
    
    # Set default font to match UI design
    font = app.font()
    font.setFamily("Times New Roman")  # Consistent with UI designer
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start application event loop
    sys.exit(app.exec())