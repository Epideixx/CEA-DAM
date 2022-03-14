from without_building import main

explosions = [(40, 25), (40,65), (55, 65), (65, 32), (82, 17)]

for explosion in explosions :

    main(folder_stations="Simulations/Simu_without_building_82_17", n = 10, explosion_source=explosion)

print('ok')