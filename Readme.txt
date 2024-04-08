######## Conteneurisation de l'api ################################

Nous avons crer une image à l'aide du fichier Dockerfile comme suit 
    Docker build -t nomimage -f Dockerfile.txt .

Ensuite nous avons lancé un conteneur a partir de l'image crer comme suit 
    Docker run -d --name conteneur1 -p 8000:8000 nomimage

