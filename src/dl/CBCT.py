from Methode.Methode import Methode

class CBCT(Methode):
    def __init__(self, widget):
        super().__init__(widget)

    def TestCheckbox(self):
        return super().TestCheckbox()

    def TestReference(self, ref_folder: str):
        return super().TestReference(ref_folder)

    def NumberScan(self, scan_folder: str):
        return super().NumberScan(scan_folder)
    
    def TestScan(self, scan_folder: str):
        return super().TestScan(scan_folder)

    def DownloadRef(self):
        return super().DownloadRef()

    def Process(self, **kwargs):
        return super().Process(kwargs)

    def DicLandmark(self):
        return super().DicLandmark()

    def ListLandmark(self):
        return super().ListLandmark()
        
    def existsLandmark(self,pathfile,pathref):
        return super().existsLandmark()

    def Sugest(self):
        return super().Sugest()

    def TestProcess(self, **kwargs) -> str:
        return super().TestProcess(kwargs)