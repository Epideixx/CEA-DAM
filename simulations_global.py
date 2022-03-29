from without_building import main
import os

explosions = [(40, 25), (40,65), (55, 65), (65, 32), (82, 17)]
folder = "Simulations"
for explosion in explosions :

    filename = "Simu_without_building_" + str(explosion[0]) + "_" + str(explosion[1])
    
    file = os.path.join(folder, filename)
    main(folder_stations=file, n = 15, explosion_source=explosion, save_file="Images/" + filename+".png")

    filename = "Simu_with_building_" + str(explosion[0]) + "_" + str(explosion[1])
    
    file = os.path.join(folder, filename)
    main(folder_stations=file, n = 15, explosion_source=explosion, save_file="Images/" + filename+".png")

print('ok')