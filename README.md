Ce code réalise le calcul de Debye en 3D et sa projection sur le plan détecteur (q_x, q_z).

La classe NanoparticleScattering2D prend en entrée un fichier .xyz et les caractéristiques expérimentales suivantes:
- longueur d'onde
- distance échantillon-detecteur
- nombre de pixels (détécteur supposé carré)
- taille de pixels

C'est donc la géométrie de l'expérience qui définit la gamme de q accessible.
Le fichier xyz utilisé en entrée peut décrire des positions atomiques ou des positions de particules dans un réseau.  
Le fichier generate_paracrystal_assembly contient des fonctions permettant de générer des assemblages de particules suivant des réseaux paracristallins.
La méthode rotate_positions() de la classe NanoparticleScattering2D permet d'orienter la particule selon le choix de l'utilisateur, suivant une rotation (alpha, beta, gamma) obéissant à la convention ZYZ des angles d'Euler. On peut ainsi orienter un réseau suivant un axe de zone défini

