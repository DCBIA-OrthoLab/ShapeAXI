import os
import logging
import time
import vtk, qt, slicer
from qt import QWidget, QVBoxLayout, QScrollArea, QTabWidget, QCheckBox
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin



class ASO(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ASO"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Automated Dental Tools"]  # set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Nathan Hutin (UoM), Luc Anchling (UoM)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
        This is an example of scripted loadable module bundled in an extension.
        See more information in <a href="https://github.com/organization/projectname#ASO">module documentation</a>.
        """
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.registerSampleData)

        #
        # Register sample data sets in Sample Data module
        #

    def registerSampleData(self):
        """
        Add data sets to Sample Data module.
        """
        # It is always recommended to provide sample data for users to make it easy to try the module,
        # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

        import SampleData
        iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

        # To ensure that the source code repository remains small (can be downloaded and installed quickly)
        # it is recommended to store data sets that are larger than a few MB in a Github release.

        # ALI1
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='ASO',
        sampleName='ASO1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'ASO1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='ASO1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='ASO1'
        )

        # ASO2
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='ASO',
        sampleName='ASO2',
        thumbnailFileName=os.path.join(iconsPath, 'ASO2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='ASO2.nrrd',
        checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='ASO2'
        )

#
# ASOWidget
#

class ASOWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initiASOzed.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False



        self.nb_patient = 0 # number of scans in the input folder






    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initiASOzed.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/ASO.ui'))
        self.layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ASOLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).


      





        """
            8b           d8                          88              88           88                         
            `8b         d8'                          ""              88           88                         
            `8b       d8'                                           88           88                         
            `8b     d8'    ,adPPYYba,  8b,dPPYba,  88  ,adPPYYba,  88,dPPYba,   88   ,adPPYba,  ,adPPYba,  
            `8b   d8'     ""     `Y8  88P'   "Y8  88  ""     `Y8  88P'    "8a  88  a8P_____88  I8[    ""  
            `8b d8'      ,adPPPPP88  88          88  ,adPPPPP88  88       d8  88  8PP"""""""   `"Y8ba,   
            `888'       88,    ,88  88          88  88,    ,88  88b,   ,a8"  88  "8b,   ,aa  aa    ]8I  
            `8'        `"8bbdP"Y8  88          88  `"8bbdP"Y8  8Y"Ybbd8"'   88   `"Ybbd8"'  `"YbbdP"'  
                                                                                                            
        """
        self.MethodeDic={'IOS':IOS(self),'CBCT':CBCT(self)}
        self.ActualMeth= Methode
        self.nb_scan = 0
        self.startprocess =0
        self.patient_process = 0
        self.dicchckbox={}  
        self.dicchckbox2={}
        """
        exemple dic = {'teeth'=['A,....],'Type'=['O',...]}
        """

        #use messletter to add big comment with univers as police










        """
                              
                                            88               88           
                                            88               ""    ,d     
                                            88                     88     
                                            88  8b,dPPYba,   88  MM88MMM  
                                            88  88P'   `"8a  88    88     
                                            88  88       88  88    88     
                                            88  88       88  88    88,    
                                            88  88       88  88    "Y888  
                              
        """
        self.initCheckbox(self.MethodeDic['IOS'],self.ui.LayoutLandmarkIOS,self.ui.tohideIOS)
        # self.initCheckbox(self.MethodeDic['CBCT'],self.ui.LayoutLandmarkCBCT,self.ui.tohideCBCT) a decommmente

















        """
                                                                                       
                        ,ad8888ba,                                                                           
                        d8"'    `"8b                                                                   ,d     
                        d8'                                                                             88     
                        88              ,adPPYba,   8b,dPPYba,   8b,dPPYba,    ,adPPYba,   ,adPPYba,  MM88MMM  
                        88             a8"     "8a  88P'   `"8a  88P'   `"8a  a8P_____88  a8"     ""    88     
                        Y8,            8b       d8  88       88  88       88  8PP"""""""  8b            88     
                        Y8a.    .a8P  "8a,   ,a8"  88       88  88       88  "8b,   ,aa  "8a,   ,aa    88,    
                        `"Y8888Y"'    `"YbbdP"'   88       88  88       88   `"Ybbd8"'   `"Ybbd8"'    "Y888  
                                                                                                            
        """

        self.ui.ButtonSearchScanLmFolder.connect('clicked(bool)',self.SearchScanLm)
        self.ui.ButtonSearchReference.connect('clicked(bool)',self.SearchReference)
        self.ui.ButtonDowloadRefLm.connect('clicked(bool)',self.DownloadRef)
        self.ui.ButtonOriented.connect('clicked(bool)',self.onPredictButton)
        self.ui.ButtonOutput.connect('clicked(bool)',self.ChosePathOutput)
        self.ui.ButtonCancel.connect('clicked(bool)',self.onCancel)
        self.ui.ButtonSugestLmIOS.clicked.connect(self.SelectSugestLandmark)
        self.ui.ButtonSugestLmCBCT.clicked.connect(self.SelectSugestLandmark)

        self.ui.CbInputType.currentIndexChanged.connect(self.SwitchType)































    """

                                                                                                                                                                    
88888888888                                                 88                                88888888ba                                                            
88                                                   ,d     ""                                88      "8b                 ,d       ,d                               
88                                                   88                                       88      ,8P                 88       88                               
88aaaaa      88       88  8b,dPPYba,    ,adPPYba,  MM88MMM  88   ,adPPYba,   8b,dPPYba,       88aaaaaa8P'  88       88  MM88MMM  MM88MMM   ,adPPYba,   8b,dPPYba,   
88           88       88  88P'   `"8a  a8"     ""    88     88  a8"     "8a  88P'   `"8a      88""""""8b,  88       88    88       88     a8"     "8a  88P'   `"8a  
88           88       88  88       88  8b            88     88  8b       d8  88       88      88      `8b  88       88    88       88     8b       d8  88       88  
88           "8a,   ,a88  88       88  "8a,   ,aa    88,    88  "8a,   ,a8"  88       88      88      a8P  "8a,   ,a88    88,      88,    "8a,   ,a8"  88       88  
88            `"YbbdP'Y8  88       88   `"Ybbd8"'    "Y888  88   `"YbbdP"'   88       88      88888888P"    `"YbbdP'Y8    "Y888    "Y888   `"YbbdP"'   88       88  
                                                                                                                                                                    
                                                                                                                                                                    

    """

    def SwitchType(self,index):
        if index == 0 :
            self.ActualMeth = self.MethodeDic['CBCT']
            self.ui.stackedWidget.setCurrentIndex(0)

        elif index == 1 :
            self.ActualMeth = self.MethodeDic['IOS']
            self.ui.stackedWidget.setCurrentIndex(1)

        self.dicchckbox=self.ActualMeth.getcheckbox()
        self.dicchckbox2=self.ActualMeth.getcheckbox2()

        self.enableCheckbox()


    def SearchScanLm(self):
        scan_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
        if not scan_folder == '':
            nb_scans = self.ActualMeth.NumberScan(scan_folder)
            error = self.ActualMeth.TestScan(scan_folder)

            if isinstance(error,str):
                qt.QMessageBox.warning(self.parent, 'Warning', error)
            else :
                self.nb_patient = nb_scans
                self.ui.lineEditScanLmPath.setText(scan_folder)
                self.ui.LabelInfoPreProc.setText("Number of scans to process : " + str(nb_scans))
                self.ui.LabelProgress.setText('Patient process : 0 /'+str(nb_scans))
                self.enableCheckbox()


                if self.ui.lineEditOutputPath.text == '':
                    dir , spl = os.path.split(scan_folder)
                    self.ui.lineEditOutputPath.setText(os.path.join(dir,spl+'Or'))

    def SearchReference(self):
        ref_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
        if not ref_folder == '':
            error = self.ActualMeth.TestReference(ref_folder)

            if isinstance(error,str):
                qt.QMessageBox.warning(self.parent, 'Warning', error)

            else:
                self.ui.lineEditRefFolder.setText(ref_folder)
                self.enableCheckbox()


    def ChosePathOutput(self):
        out_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
        if not out_folder =='':
            self.ui.lineEditOutputPath.setText(out_folder)

    def DownloadRef(self):
        self.ActualMeth.DownloadRef()


    def SelectSugestLandmark(self):
        best = self.ActualMeth.Sugest()
        for listcheckbox in self.dicchckbox.values():
            for checkbox in listcheckbox:
                if checkbox.text in best and checkbox.isEnabled():
                    checkbox.setCheckState(True)
                else :
                    checkbox.setCheckState(False)


    
    def enableCheckbox(self):
        status = self.ActualMeth.existsLandmark(self.ui.lineEditScanLmPath.text,self.ui.lineEditRefFolder.text)
        if status is None:
            return
        for checkboxs,checkboxs2 in zip(self.dicchckbox.values(),self.dicchckbox2.values()):
            for checkbox, checkbox2 in zip(checkboxs,checkboxs2):
                checkbox.setEnabled(status[checkbox.text])
                checkbox2.setEnabled(status[checkbox2.text])


























    """
                                                                                    
                        88888888ba                                                                          
                        88      "8b                                                                         
                        88      ,8P                                                                         
                        88aaaaaa8P'  8b,dPPYba,   ,adPPYba,    ,adPPYba,   ,adPPYba,  ,adPPYba,  ,adPPYba,  
                        88""""""'    88P'   "Y8  a8"     "8a  a8"     ""  a8P_____88  I8[    ""  I8[    ""  
                        88           88          8b       d8  8b          8PP"""""""   `"Y8ba,    `"Y8ba,   
                        88           88          "8a,   ,a8"  "8a,   ,aa  "8b,   ,aa  aa    ]8I  aa    ]8I  
                        88           88           `"YbbdP"'    `"Ybbd8"'   `"Ybbd8"'  `"YbbdP"'  `"YbbdP"'  
                                                                                
                                                                                    
    """





    def onPredictButton(self):
        error = self.ActualMeth.TestProcess(input_folder = self.ui.lineEditScanLmPath.text, gold_folder = self.ui.lineEditRefFolder.text,
                                        folder_output = self.ui.lineEditOutputPath.text, add_in_namefile = self.ui.lineEditAddName.text, dic_checkbox = self.dicchckbox,
                                        label_surface = self.ui.lineEditLabelSurface.text)


        if isinstance(error,str):
            qt.QMessageBox.warning(self.parent, 'Warning',error.replace(',','\n'))

        else :
            self.process = self.ActualMeth.Process(input_folder = self.ui.lineEditScanLmPath.text, gold_folder = self.ui.lineEditRefFolder.text,
                                        folder_output = self.ui.lineEditOutputPath.text, add_in_namefile = self.ui.lineEditAddName.text, dic_checkbox = self.dicchckbox,
                                        label_surface = self.ui.lineEditLabelSurface.text)

            self.processObserver = self.process.AddObserver('ModifiedEvent',self.onProcessUpdate)
            self.onProcessStarted()



    def onProcessStarted(self):
        self.startTime = time.time()

        self.ui.progressBar.setMaximum(self.nb_patient)
        self.ui.progressBar.setValue(0)


        self.ui.LabelProgress.setText(f"Patient : 0 / {self.nb_patient}")

        self.nb_patient_treat = 0
        self.progress = 0

        self.RunningUI(True)



    def onProcessUpdate(self,caller,event):

        timer = f"Time : {time.time()-self.startTime:.2f}s"
        self.ui.LabelTimer.setText(timer)
        progress = caller.GetProgress()

        if progress == 0:
            self.updateProgessBar = False

        if progress != 0 and self.updateProgessBar == False:
            self.updateProgessBar = True
            self.nb_patient_treat+=1
            self.ui.progressBar.setValue(self.nb_patient_treat)
            self.ui.LabelProgress.setText(f"patient : {self.nb_patient_treat} / {self.nb_patient}")

        if self.process.GetStatus() & self.process.Completed:
            # process complete


            if self.process.GetStatus() & self.process.ErrorsMask:
                # error
                print("\n\n ========= PROCESSED ========= \n")

                print(self.process.GetOutputText())
                print("\n\n ========= ERROR ========= \n")
                errorText = self.process.GetErrorText()
                print("CLI execution failed: \n \n" + errorText)


            else:


                self.OnEndProcess()


    def OnEndProcess(self):

        
        print('PROCESS DONE.')
        self.RunningUI(False)

        stopTime = time.time()

        logging.info(f'Processing completed in {stopTime-self.startTime:.2f} seconds')


    def onCancel(self):


        self.process.Cancel()


        self.RunningUI(False)




    def RunningUI(self, run = False):

        self.ui.ButtonOriented.setVisible(not run)

      
        self.ui.progressBar.setVisible(run)
        self.ui.LabelTimer.setVisible(run)





























    """
                                                                                                                        
   ad88                                                 88                                88               88           
  d8"                                            ,d     ""                                88               ""    ,d     
  88                                             88                                       88                     88     
MM88MMM  88       88  8b,dPPYba,    ,adPPYba,  MM88MMM  88   ,adPPYba,   8b,dPPYba,       88  8b,dPPYba,   88  MM88MMM  
  88     88       88  88P'   `"8a  a8"     ""    88     88  a8"     "8a  88P'   `"8a      88  88P'   `"8a  88    88     
  88     88       88  88       88  8b            88     88  8b       d8  88       88      88  88       88  88    88     
  88     "8a,   ,a88  88       88  "8a,   ,aa    88,    88  "8a,   ,a8"  88       88      88  88       88  88    88,    
  88      `"YbbdP'Y8  88       88   `"Ybbd8"'    "Y888  88   `"YbbdP"'   88       88      88  88       88  88    "Y888  
                                                                                                                        
                                                                                                                        
    """


    def initCheckbox(self,methode,layout,tohide : qt.QLabel):
        tohide.setHidden(True)
        dic  = methode.DicLandmark()
        # status = methode.existsLandmark('','')
        dicchebox={}
        dicchebox2={}
        for type , tab in dic.items():
            
            Tab = QTabWidget()
            layout.addWidget(Tab)
            listcheckboxlandmark =[]
            listcheckboxlandmark2 = []

            all_checkboxtab = self.CreateMiniTab(Tab,'All',0)
            for i, (name , listlandmark) in enumerate(tab.items()):
                widget = self.CreateMiniTab(Tab,name,i+1)
                for landmark in listlandmark:
                    checkbox  = QCheckBox()
                    checkbox2 = QCheckBox()
                    checkbox.setText(landmark)
                    checkbox2.setText(landmark)
                    # checkbox.setEnabled(status[landmark])
                    # checkbox2.setEnabled(status[landmark])
                    checkbox2.toggled.connect(checkbox.setChecked)
                    checkbox.toggled.connect(checkbox2.setChecked)
                    widget.addWidget(checkbox)
                    all_checkboxtab.addWidget(checkbox2)
                    
                    listcheckboxlandmark.append(checkbox)
                    listcheckboxlandmark2.append(checkbox2)
                    

            dicchebox[type] = listcheckboxlandmark
            dicchebox2[type]=listcheckboxlandmark2

        methode.setcheckbox(dicchebox)
        methode.setcheckbox2(dicchebox2)



    def CreateMiniTab(self,tabWidget : QTabWidget, name : str, index : int):
    

        new_widget = QWidget()
        new_widget.setMinimumHeight(250)

        layout = QVBoxLayout(new_widget)

        scr_box = QScrollArea(new_widget)
        scr_box.setMinimumHeight(200)

        layout.addWidget(scr_box)

        new_widget2 = QWidget(scr_box)
        layout2 = QVBoxLayout(new_widget2)

        
        layout.addStretch()
        scr_box.setWidgetResizable(True)
        scr_box.setWidget(new_widget2)

        
        tabWidget.insertTab(index,new_widget,name)

        return layout2

    




























    """
                                                                                            
                                ,ad8888ba,             88                                   
                                d8"'    `"8b     ,d     88                                   
                                d8'        `8b    88     88                                   
                                88          88  MM88MMM  88,dPPYba,    ,adPPYba,  8b,dPPYba,  
                                88          88    88     88P'    "8a  a8P_____88  88P'   "Y8  
                                Y8,        ,8P    88     88       88  8PP"""""""  88          
                                Y8a.    .a8P     88,    88       88  "8b,   ,aa  88          
                                `"Y8888Y"'      "Y888  88       88   `"Ybbd8"'  88          
    """

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        if self.logic.cliNode is not None:
            # if self.logic.cliNode.GetStatus() & self.logic.cliNode.Running:
            self.logic.cliNode.Cancel() 

        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # if inputParameterNode:
        self.setParameterNode(self.logic.getParameterNode())

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # Update buttons states and tooltips
        # if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
        #   self.ui.applyButton.toolTip = "Compute output volume"
        #   self.ui.applyButton.enabled = True
        # else:
        #   self.ui.applyButton.toolTip = "Select input and output volume nodes"
        #   self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)











#
# ASOLogic
#

class ASOLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        self.cliNode = None


    def process(self, parameters):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded

        """

        # import time
        # startTime = time.time()

        logging.info('Processing started')

        PredictProcess = slicer.modules.aso_ios


        self.cliNode = slicer.cli.run(PredictProcess, None, parameters)


        return PredictProcess



























