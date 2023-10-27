import saxi_transforms
import torch
import utils

if __name__ == "__main__":
    # # Create a PyTorch tensor from your data
    # surf = torch.randn(10, 3)

    # unit_surf_transform = saxi_transforms.UnitSurfTransform()
    # unit_surf = unit_surf_transform(surf)

    # random_surf_transform = saxi_transforms.RandomRotation()
    # random_surf = random_surf_transform(surf)

    # print(surf)

    # print("UnitSurfTransform")
    # print(unit_surf)

    # print("RandomRotation")
    # print(random_surf)

    coucou = utils.is_vtk_file("/work/floda/data/DCBIA/DJD/Vtks_Non_Oriented/Vtk_Controls_per_patients/r_10/T1/r_10_T1_Left_condyle.vtk")
    if coucou == True:
        print("coucou")
