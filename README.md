Ce code réalise le calcul de Debye en 3D et sa projection sur le plan détecteur (q_x, q_z).

La classe NanoparticleScattering2D prend en entrée un fichier xyz et en calcule la diffusion théorique sur la gamme de q souhaitée.
La méthode rotate_positions() permet d'orienter la particule selon le choix de l'utilisateur, suivant une rotation (alpha, beta, gamma) obéissant à la convention ZYZ des angles d'Euler.

Il reste à implémenter le moyenne orientationnelle (lebedev, fibonacci ou gauss Legendre), et à comparer  ce calcul avec un calcul réalisé par Debyecalculator
