import pandas as pd
import matplotlib.pyplot as plt

from load_data import loadDataset


class LoadDataset(loadDataset):
    def _Init_(dataframe,img_path:str,
               x_col:str,y_col:str,
               box_col:list,
               subset:str=None,split:tuple=None):
        pass
    def Len(self)->int:
        return self.__len__()
        
    def Get_dataframe(self)->pd.DataFrame:
        return self.get_dataframe()
    
    def Label_(self)->dict:
        return self.label_
    
    def Plot_image(self,index:int,line_thickness:int=5)->plt.imshow:
        return self.plot_image(index,line_thickness=line_thickness)
    
    def Multi_plot_image(self,line_thickness : int =5)->plt.imshow:
        return self.multi_plot_image(line_thickness=line_thickness)

    def Dataloader(self):
        return self.dataloader()