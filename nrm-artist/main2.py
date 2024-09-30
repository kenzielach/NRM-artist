from masks_functs import *
from design_class import *

aperture = pyfits.getdata('/Users/kenzie/Desktop/CodeAstro/planet-guts/keck_aperture.fits') # set Keck primary aperture

#print('Hello! Please enter the number of holes for your mask:')
#nholes = int(input())
#check_valid_nholes(nholes)
nholes = 9

#print('Please enter the hole radius in meters:')
#hrad = float(input())
#check_valid_hrad(hrad)
hrad = 0.65

print('Thanks! Generating mask design...')

i = 0

if nholes > 6:
    while 1:
        mask_design, vcoords = make_design(6, hrad, return_vcoords=True)
        diff = nholes - 6
        mask_design = add_to_design(mask_design, diff, vcoords)   
        if mask_design.nholes == nholes:
            mask_design.uv_coords = np.empty([x_choose_y(nholes, 2), 2])
            mask_design.get_uvs()
            stuff = plot_ps(mask_design)    
            parameters, covariance, fit, std1, std2, ar, cent, sum21 = fit_ps_fft(stuff)
            print('symmetry value: ' + str(ar))
            print('compactness value: ' + str(sum21) + '\n')
            #if ar < 50.08 and sum21 < 5450: # for 6 hole
            #if ar < 0.23 and sum21 < 1900: # for 7 hole
            if ar < 0.1 and sum21 < 11000: # for 9 hole
                test = design(9, 0.65)
                test.xy_coords_m = np.loadtxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/9hole_nirc2_xycoords_cent.txt')
                test.xy_coords_cm = 100 * test.xy_coords_m
                test.get_uvs()
                stuff2 = plot_ps(test)
                p1, p2, p3, p4, p5, p6, p7, sum22 = fit_ps_fft(stuff2)
                print('the best-fit symmetry/compactness values are: ' + str(ar) + '/' + str(sum21))
                print('the NIRC2 array symmetry/compactness values are: ' + str(p6) + '/' + str(sum22))
                coords = np.append(mask_design.uv_coords, -mask_design.uv_coords, axis=0)
                flatx = np.sort(coords[:, 0])
                flaty = np.sort(coords[:, 1])

                plt.figure()
                plt.title('Power spectrum')
                plt.imshow(stuff)

                plt.figure()
                plt.title('zoomed in FFT')
                plt.imshow(cent)
                plt.colorbar()

                plt.figure()
                plt.imshow(fit.reshape([len(cent), len(cent)]))
                plt.colorbar()
                plt.title('best fit (optimized)')

                plt.figure()
                plt.imshow(p3.reshape([len(p7), len(p7)]))
                plt.colorbar()
                plt.title('best fit (manx)')

                plt.figure()
                plt.imshow(fit.reshape([len(cent), len(cent)]) - p3.reshape([len(p7), len(p7)]))
                plt.colorbar()
                plt.title('optimized - manx')

                print('plot aperture? y/n')
                flag = input()
                if flag == 'y':
                    plot_design(mask_design, aperture)
                break
else:
    while 1:
        mask_design = make_design(nholes, hrad)
        stuff = plot_ps(mask_design)    
        parameters, covariance, fit, std1, std2, ar, cent, sum21 = fit_ps_fft(stuff)
        print('symmetry value: ' + str(ar))
        print('compactness value: ' + str(sum21) + '\n')
        if ar < 50.08 and sum21 < 5450:
            test = design(7, 0.65)
            test.xy_coords_m = np.loadtxt('/Users/kenzie/Desktop/Laniakea/Finalized_mask_pipeline/masks/SPIE_2024/7hole_asym_xycoords_cent.txt')
            test.xy_coords_cm = 100 * test.xy_coords_m
            test.get_uvs()
            stuff2 = plot_ps(test)
            p1, p2, p3, p4, p5, p6, p7, sum22 = fit_ps_fft(stuff2)
            print('the best-fit symmetry/compactness values are: ' + str(ar) + '/' + str(sum21))
            print('the SPHERE array symmetry/compactness values are: ' + str(p6) + '/' + str(sum22))
            coords = np.append(mask_design.uv_coords, -mask_design.uv_coords, axis=0)
            flatx = np.sort(coords[:, 0])
            flaty = np.sort(coords[:, 1])

            plt.figure()
            plt.title('Power spectrum')
            plt.imshow(stuff)

            plt.figure()
            plt.title('zoomed in FFT')
            plt.imshow(cent)
            plt.colorbar()

            plt.figure()
            plt.imshow(fit.reshape([len(cent), len(cent)]))
            plt.colorbar()
            plt.title('best fit (optimized)')

            plt.figure()
            plt.imshow(p3.reshape([len(p7), len(p7)]))
            plt.colorbar()
            plt.title('best fit (manx)')

            plt.figure()
            plt.imshow(fit.reshape([len(cent), len(cent)]) - p3.reshape([len(p7), len(p7)]))
            plt.colorbar()
            plt.title('optimized - manx')

            print('plot aperture? y/n')
            flag = input()
            if flag == 'y':
                plot_design(mask_design, aperture)
             
            break