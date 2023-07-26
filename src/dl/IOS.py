from Methode.Methode import Methode
import slicer
import webbrowser
import glob

class IOS(Methode):
    def __init__(self, widget):
        super().__init__(widget)


    def NumberScan(self, scan_folder: str):
            
        return len(super().search(scan_folder,'vtk'))


    def TestScan(self, scan_folder: str):
        out = None
        if self.NumberScan(scan_folder) == 0 :
            out = 'Give folder with vkt file'
        return out


    def TestReference(self, ref_folder: str):
        list = glob.glob(ref_folder+'/*vtk')
        out = None
        if len(list) == 0:
            out = 'Please choose a folder with json file'
        elif len(list)>2:
            out = 'Too many json file '
        return out

    def TestCheckbox(self,dic_checkbox) -> str:
        list_landmark = self.__CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_landmark.split(','))< 3:
             out = 'Give minimum 3 landmark'
        return out


    def DownloadRef(self):
        webbrowser.open('https://github.com/HUTIN1/ASO/releases/tag/v1.0.0')


        

    def TestProcess(self,**kwargs) -> str:
        out  = ''

        scan = self.TestScan(kwargs['input_folder'])
        if isinstance(scan,str):
            out = out + f'{scan},'

        reference =self.TestReference(kwargs['gold_folder'])
        if isinstance(reference,str):
            out = out + f'{reference},'

        if kwargs['folder_output'] == '':
            out = out + "Give output folder,"

        testcheckbox = self.TestCheckbox(kwargs['dic_checkbox'])
        if isinstance(testcheckbox,str):
            out = out + f"{testcheckbox},"

        if kwargs['add_in_namefile']== '':
            out = out + "Give something to add in name of file,"

        if kwargs['label_surface'] == '':
            out = out + "Give Label Surface"

        if out != '':
            out=out[:-1]

        else : 
            out = None

        return out


    def Process(self, **kwargs):
        list_teeth = self.__CheckboxisChecked(kwargs['dic_checkbox'])
        print('label',kwargs['label_surface'])

        parameter= {'input':kwargs['input_folder'],'gold_folder':kwargs['gold_folder'],'output_folder':kwargs['folder_output'],'add_inname':kwargs['add_in_namefile'],'list_teeth':list_teeth ,'label_surface':kwargs['label_surface']}


        print('parameter',parameter)
        OrientProcess = slicer.modules.aso_ios
        process = slicer.cli.run(OrientProcess,None,parameter)

        return process

    def DicLandmark(self):
       
        dic = {'Teeth':
                    {'Upper':['2','3','4','5','6','7','8','9','10','11','12','13','14',',15'],
                    'Lower':['16','17','18','19',',20','21','22','23','24','25','26','27','28','29','30','31','32']
                    },
                }
        return dic





    def existsLandmark(self,folderpath,reference_folder):

        return None


    def Sugest(self):
        return ['4','9','13','20','25','30']


    def __CheckboxisChecked(self,diccheckbox : dict):
        out=''
        if not len(diccheckbox) == 0:

            for checkboxs in diccheckbox.values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        out+=f'{checkbox.text},'
            while out[0]==',':
                out = out[1:]

            before = None
            for i, letter in enumerate(out):
                if before==',' and letter==',':
                    out = out[:i]+out[i+1:]
                before = letter
                
            out=out[:-1]
        print('out',out)
        return out
