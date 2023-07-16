#GPU下进行预测
import numpy as np 
import SimpleITK as sitk
from paddle.inference import create_predictor,Config

class Predictor:
    """
    用于预测的类
    """
    def __init__(self,model_path,param_path):
        self.pred_cfg = Config(model_path,param_path)
        self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)
        # self.pred_cfg.enable_use_gpu(100, 0)
        self.pred_cfg.disable_gpu()
        self.predictor = create_predictor(self.pred_cfg)

    def predict(self, data):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        self.predictor.run()
        result = output_handle.copy_to_cpu()
        return result

def resampleImage(sitkimg,new_shape,new_spacing):
    #对SimpleITK 的数据进行重新采样。重新设置spacing和shape
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkimg)  
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_shape)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(sitkimg)  

def crop_wwwc(sitkimg,max_v,min_v):
    #对SimpleITK的数据进行窗宽窗位的裁剪，应与训练前对数据预处理时一致
    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMaximum(max_v)
    intensityWindow.SetWindowMinimum(min_v)
    return intensityWindow.Execute(sitkimg)

def GetLargestConnectedCompont(binarysitk_image):
    # 最大连通域提取,binarysitk_image 是掩膜
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)#根据掩膜计算统计量
    # stats.
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels(): # 掩膜中存在的标签类别
        size = stats.GetPhysicalSize(l)
        if maxsize < size: # 只保留最大的标签类别
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    if len(stats.GetLabels()):
        outmask[labelmaskimage == maxlabel] = 255
        outmask[labelmaskimage != maxlabel] = 0
    return outmask

LABLES = {0:"background",
    1:"spleen",
    2:"right kidney",
    3:"left kidney",
    4:"gall bladder",
    5:"esophagus",
    6:"liver",
    7:"stomach",
    8:"arota",
    9:"postcava",
    10:"pancreas",
    11:"bladder"
    }

def predict_nii(origin_path):

    import nibabel as nib
    import matplotlib.pyplot as plt

    image_obj = nib.load(origin_path)
    image_data = image_obj.get_fdata()
    X, Y, Z = image_data.shape
    plt.imshow(image_data[:, :, int(Z/2)], cmap='gray',aspect='auto')
    plt.savefig(origin_path+'_Z.png')
    plt.imshow(image_data[int(X/2), :, :], cmap='gray',aspect='auto')
    plt.savefig(origin_path+'_X.png')
    plt.imshow(image_data[:, int(Y/2), :], cmap='gray',aspect='auto')
    plt.savefig(origin_path+'_Y.png')    
    
    origin = sitk.ReadImage(origin_path)

    new_shape = (128, 128, 128) #xyz #这个形状与训练的对数据预处理的形状要一致
    image_shape = origin.GetSize()
    spacing = origin.GetSpacing()
    new_spacing = tuple((image_shape / np.array(new_shape)) *spacing) 

    itk_img_res = resampleImage(origin,new_shape,new_spacing)  # 得到重新采样后的图像
    itk_img_res = crop_wwwc(itk_img_res,max_v=611,min_v=-338)#和预处理文件一致
    npy_img = sitk.GetArrayFromImage(itk_img_res).astype("float32")
    input_data = np.expand_dims(npy_img,axis=0)
    if input_data.max() > 0: #归一化
        input_data = input_data / input_data.max()
    input_data = np.expand_dims(input_data,axis=0)
    print(f"输入网络前数据的形状:{input_data.shape}")#shape(1, 1, 128, 128, 256)

    #创建预测器，加载模型进行预测
    predictor = Predictor('output/model.pdmodel',
                            'output/model.pdiparams')
    output_data = predictor.predict(input_data)
    print(f"预测结果的形状：{output_data.shape}")#shape (1, 128, 128, 256)

    #加载3d模型预测的mask，由numpy 转换成SimpleITK格式
    data = np.squeeze(output_data)
    mask_itk_new = sitk.GetImageFromArray(data)
    mask_itk_new.SetSpacing(new_spacing)
    mask_itk_new.SetOrigin(origin.GetOrigin())
    mask_itk_new.SetDirection(origin.GetDirection())
    mask_itk_new = sitk.Cast(mask_itk_new,sitk.sitkUInt8)

    x,y,z = mask_itk_new.GetSize()
    mask_array = np.zeros((z,y,x),np.uint8)
    max_value = np.max(sitk.GetArrayViewFromImage(mask_itk_new))
    #对转换成SimpleITK的预测mask进行处理，只保留最大连通域，去除小目标
    for index in range(1,max_value+1):
        sitk_seg = sitk.BinaryThreshold(mask_itk_new, lowerThreshold=index, upperThreshold=index, insideValue=255, outsideValue=0)
        # step2.形态学开运算
        BMO = sitk.BinaryMorphologicalOpeningImageFilter()
        BMO.SetKernelType(sitk.sitkNearestNeighbor)
        BMO.SetKernelRadius(2)
        BMO.SetForegroundValue(1)
        sitk_open = BMO.Execute(sitk_seg!=0)
        #提取每个椎体的最大连通域提取，为了去掉小目标
        sitk_open_array = GetLargestConnectedCompont(sitk_open)
        mask_array[sitk_open_array==255] = int(index)

    #对处理好的预测mask，重采样原始的size 和spacing
    sitkMask = sitk.GetImageFromArray(mask_array)
    sitkMask.CopyInformation(mask_itk_new)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkMask)  # 需要重新采样的目标图像
    resampler.SetSize(origin.GetSize())
    resampler.SetOutputSpacing(origin.GetSpacing())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    Mask = resampler.Execute(sitkMask)  # 得到重新采样后的图像
    Mask.CopyInformation(origin)
    sitk.WriteImage(Mask, origin_path.replace('.nii','_predict_v2.nii'))
    print("预测成功！")

    import vtk
    filename_nii = origin_path.replace('.nii','_predict_v2.nii')
    filename = filename_nii.split(".")[0]
    results = {}
    
    label_obj = nib.load(filename_nii)
    label_array = label_obj.get_fdata()
    plt.imshow(label_array[:, :, int(Z/2)],aspect='auto')
    plt.savefig(filename_nii+'_Z.png',)
    plt.imshow(label_array[int(X/2), :, :],aspect='auto')
    plt.savefig(filename_nii+'_X.png')
    plt.imshow(label_array[:, int(Y/2), :],aspect='auto')
    plt.savefig(filename_nii+'_Y.png')
    
    if True:
        multi_label_image=sitk.ReadImage(filename_nii)
        img_npy = sitk.GetArrayFromImage(multi_label_image)
        labels = np.unique(img_npy)
        
        # read the file
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filename_nii)
        reader.Update()
        
        # for all labels presented in the segmented file
        for label in labels:

            if int(label) != 0:

                # apply marching cube surface generation
                surf = vtk.vtkDiscreteMarchingCubes()
                surf.SetInputConnection(reader.GetOutputPort())
                surf.SetValue(0, int(label)) # use surf.GenerateValues function if more than one contour is available in the file
                surf.Update()
                
                #smoothing the mesh
                smoother= vtk.vtkWindowedSincPolyDataFilter()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    smoother.SetInput(surf.GetOutput())
                else:
                    smoother.SetInputConnection(surf.GetOutputPort())
                
                # increase this integer set number of iterations if smoother surface wanted
                smoother.SetNumberOfIterations(30) 
                smoother.NonManifoldSmoothingOn()
                smoother.NormalizeCoordinatesOn() #The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
                smoother.GenerateErrorScalarsOn()
                smoother.Update()
                
                # save the output
                writer = vtk.vtkSTLWriter()
                writer.SetInputConnection(smoother.GetOutputPort())
                writer.SetFileTypeToASCII()
                
                # file name need to be changed
                # save as the .stl file, can be changed to other surface mesh file
                writer.SetFileName(f'{filename}_{label}.obj')
                writer.Write()
                results[LABLES[label]] = f'{filename}_{label}.obj'
                results[LABLES[label]+'_S'] = str((img_npy == label).sum())
    return results
                
if __name__ == '__main__':
    origin_path = 'results-0709/0f593c1e-4bb8-470f-a87b-fee3dbd3b3ed.nii'
    results = predict_nii(origin_path)
    print(results)
    