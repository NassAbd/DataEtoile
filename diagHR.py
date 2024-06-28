## Diagramme HR

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Importer les modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Charger les données du fichier CSV
data = pd.read_csv('hipparcos_data.csv')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Filtrer les colonnes nécessaires
data = data[['HIP', 'Vmag', 'Plx', 'B-V', 'SpType']]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Remplacer les valeurs NaN dans la colonne 'SpType' par une chaîne vide
data['SpType'].fillna('', inplace=True)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Filtrer les étoiles avec une parallaxe positive et significative
data = data[data['Plx'] > 0]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Calcul de la distance (en pc) 
data['Distance'] = 1/data['Plx'] * 1000   # convertir la distance en pc (convertir la parallaxe (mas) en as)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Calcul de la magnitude absolue 
data['AbsoluteMag'] = data['Vmag'] - 5 * np.log10(data['Distance'] / 10)   


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Fonction pour calculer la température effective des étoiles
def bv_to_temp(bv):
    return 4600 * ((1 / (0.92 * bv + 1.7)) + (1 / (0.92 * bv + 0.62)))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Ajouter la température aux données
data['Temperature'] = bv_to_temp(data['B-V'])


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Enregistrer le DataFrame dans un fichier CSV
data.to_csv('data.csv', index=False)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Définir les étiquettes de température et les types spectraux
temp_values = [50000, 30000, 10000, 7000, 5500, 4500, 3500]              # Étiquettes des températures
spectral_types_positions = [-0.25, -0.1, 0.1, 0.3, 0.6, 1.0, 1.7]        # Pour la position des étiquettes du type spectral
spectral_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']                     # Étiquettes du type spectral


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Définir un fond noir uniforme
plt.style.use('dark_background')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Tracer le diagramme HR
fig, ax1 = plt.subplots(figsize=(12, 8))

sc = ax1.scatter(data['B-V'], data['AbsoluteMag'], s=0.4, c=data['Temperature'], cmap='plasma')
cb = plt.colorbar(sc, label='Température (K)')
ax1.invert_yaxis()  # Inverser l'axe des magnitudes pour correspondre à la convention HR
ax1.set_xlabel('Couleur (B-V)')
ax1.set_ylabel('Magnitude Absolue')
ax1.set_xlim(-0.5, 2.0)
ax1.set_title('Diagramme de Hertzsprung-Russell')



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Ajouter l'axe supérieur pour la température
ax2 = ax1.twiny()                                        # Création d'un nouvel axe 'ax2' qui partage le même axe des ordonnées avec 'ax1'
ax2.set_xlim(ax1.get_xlim())                             # Assurer que les limites de l'axe des abscisses de ax2 correspondent à celles de ax1 pour un alignement correct
ax2.set_xticks([-0.5, -0.25, 0.0, 0.3, 0.6, 1.0, 1.5])   # Pour la position des étiquettes de température
ax2.set_xticklabels(temp_values)                         # Définir les étiquettes des graduations sur l'axe des abscisses de ax2 avec les valeurs de température
ax2.set_xlabel('Température (K)')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Ajouter un deuxième axe pour les types spectraux
ax3 = ax1.twiny()
ax3.set_xticks(spectral_types_positions)
ax3.set_xticklabels(spectral_types[:len(spectral_types_positions)])
ax3.xaxis.set_ticks_position('bottom')                  # Placer les graduations de l'axe des abscisses de ax3 en bas du graphique
ax3.xaxis.set_label_position('bottom')                  # Positionner l'étiquette de l'axe des abscisses de ax3 en bas du graphique
ax3.set_xlabel('Type Spectral')
ax3.spines['bottom'].set_position(('outward', 36))      # Ajustement de la position vers le bas
ax3.set_xlim(ax1.get_xlim())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Courbe ajustée de la séquence principale
def plot_sequence_principale(ax):
    bv_seq_princ = np.linspace(-0.2, 1.7, 100)                                        # Générer des valeurs de B-V couvrant la séquence principale
    abs_mag_seq_princ = 4.95 * bv_seq_princ**3 - 12.07 * bv_seq_princ**2 + 13.91 * bv_seq_princ - 0 # Calculer la magnitude absolue correspondante pour chaque valeur de BV 
    ax.plot(bv_seq_princ, abs_mag_seq_princ, color='yellow', linestyle='--', linewidth=2, label='Séquence Principale') # Tracer la séquence principale sur le graphique

plot_sequence_principale(ax1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Courbe ajustée de la branche des géantes rouges 
def plot_geantes(ax):
    bv_geantes = np.linspace(0.15, 1.7, 100)
    abs_mag_geantes = -2.1 * bv_geantes**2.2 + 2.9 * bv_geantes + 0.5
    ax.plot(bv_geantes, abs_mag_geantes, color='red', linestyle='--', linewidth=2, label='Géantes')

plot_geantes(ax1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Courbe ajustée de la branche des supergéantes
def plot_supergiantes(ax):
    bv_superg = np.linspace(-0.2, 1.7, 100)
    abs_mag_superg = - 6 * np.exp(-bv_superg) - 1.2 * bv_superg**2 - 1
    ax.plot(bv_superg, abs_mag_superg, color='orange', linestyle='--', linewidth=2, label='Supergéantes')

plot_supergiantes(ax1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Courbe ajustée de la branche des naines blanches
def plot_naines_blanches(ax):
    bv_naine_blanche = np.linspace(-0.2, 1.0, 100)
    abs_mag_naine_blanche = -2.1 * (bv_naine_blanche + 0.5)**2 + 5.9 * (bv_naine_blanche + 0.5) + 9
    ax.plot(bv_naine_blanche, abs_mag_naine_blanche, color='cyan', linestyle='--', linewidth=2, label='Naines Blanches')

plot_naines_blanches(ax1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Afficher la légende en haut à gauche
ax1.legend(loc='upper left')
plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Carte de densité 2D du diagramme HR

# Définir la taille de la figure
plt.figure(figsize=(10, 8))

# Créer une carte de densité 2D avec les données 'B-V' et 'AbsoluteMag'
sns.kdeplot(
    x=data['B-V'],              # Axe x correspondant à la colonne 'B-V' des données
    y=data['AbsoluteMag'],      # Axe y correspondant à la colonne 'AbsoluteMag' des données
    cmap="plasma",              # Utiliser la palette de couleurs 'plasma' pour la densité
    fill=True,                  # Remplir les contours de la densité pour une meilleure visualisation
    cbar=True                   # Ajouter une barre de couleur pour indiquer l'échelle de densité
)

# Inverser l'axe y pour que les valeurs plus élevées soient en bas, comme dans un diagramme H-R typique
plt.gca().invert_yaxis()

# Ajouter un label à l'axe x
plt.xlabel('Couleur (B-V)')

# Ajouter un label à l'axe y
plt.ylabel('Magnitude Absolue')

# Ajouter un titre à la figure
plt.title('Carte de densité 2D du diagramme H-R')

# Ajouter une limite à l'axe des abscisses
plt.xlim(-0.5, 2)

# Afficher la figure
plt.show()






# Données des isochrones (en utilisant des modèles comme ceux de MIST, PARSEC, etc.)
isochrones = {
    '1 Gyr': {'B-V': [0.1, 0.5, 0.9], 'Magnitude': [4, 2, 0]},
    '5 Gyr': {'B-V': [0.2, 0.6, 1.0], 'Magnitude': [5, 3, 1]},
    '10 Gyr': {'B-V': [0.3, 0.7, 1.1], 'Magnitude': [6, 4, 2]}
}

# Tracer le diagramme HR avec les isochrones
plt.figure(figsize=(10, 8))
plt.scatter(df['B-V'], df['Magnitude'], color='blue', s=10, label='Données observées')
for age, data in isochrones.items():
    plt.plot(data['B-V'], data['Magnitude'], label=f'Isochrone {age}')
plt.gca().invert_yaxis()
plt.xlabel('B-V')
plt.ylabel('Magnitude')
plt.legend()
plt.title('Diagramme HR avec Isochrones')
plt.show()
