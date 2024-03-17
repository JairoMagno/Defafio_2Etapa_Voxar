"""
Autor: Jairo Magno Caracciolo Marques
Descrição: Código para a 2° Etapa do processo seletivo Voxar.
"""

# Este código faz parte da segunda etapa do processo seletivo da Voxar.
# Ele é responsável por visualizar poses em um vídeo usando o modelo SimpleHigherHRNet.
# O código permite selecionar entre diferentes modos de visualização (Coco, mpii ou crowdpose)
# e exibe as articulações detectadas do vídeo selecionado.

from SimpleHigherHRNet import SimpleHigherHRNet
from misc.visualization import joints_dict, draw_points_and_skeleton
from cv2 import (VideoCapture, imshow, waitKey, destroyAllWindows, resize, imshow, imread, IMREAD_COLOR, circle, line,
                 CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS)

# Varíaveis constantes para uso.
WIDTH : int = 720    # Largura em Pixels
HEIGHT: int = 480    # Altura em Pixels

def test_single_image():
    model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
    image = imread("image.jpg", IMREAD_COLOR)

    joints = joints_dict()['coco']['skeleton']
    points = model.predict(image)
    for _, point in enumerate(points):
        image = draw_points_and_skeleton(image, point, joints)

    imshow('Teste', image)

    waitKey(0) 
    # Fechando todas as janelas abertas 
    destroyAllWindows()

def select_pose():
    """
    Função para selecionar o modo de visualização.
    """
    print(31 * "=")
    print("Escolha o modo de visualização:")
    print("1 - Visualização de Coco")
    print("2 - Visualização de mpii")
    print("3 - Visualização de crowdpose")
    print(31 * "=")

    try:
        user_input = int(input())
    except ValueError:
        raise ValueError('Formato Inválido! Escolha 1, 2 ou 3')
    
    if user_input == 1:
        total_points = 17
        pose = 'coco'
    elif user_input == 2:
        total_points = 15
        pose = 'mpii'
    elif user_input == 3:
        total_points = 14
        pose = 'crowdpose'
    else:
        raise ValueError('Valor selecionado inválido! Escolha 1, 2 ou 3')

    return total_points, pose

def main():
    total_points, pose = select_pose()

    # Possíveis pesos para utilização
    weight1: str = "./weights/pose_higher_hrnet_w48_640.pth"
    weight2: str = "./weights/pose_higher_hrnet_w32_640.pth"
    weight3: str = "./weights/pose_higher_hrnet_w32_512.pth"

    model = SimpleHigherHRNet(32, total_points, weight3, resolution=100)
    capture = VideoCapture("video/panoptic.mp4")   #Objeto OpenCV capaz de capturar ou visualizar vídeos.

    # Checa se a conexão com a câmera foi estabelecida
    if not capture.isOpened():
        print("Houve um erro na abertura do arquivo!")
    
    else:
        # Pega as propriedades do vídeo e printa
        frame_width: int = capture.get(CAP_PROP_FRAME_WIDTH)
        frame_height: int = capture.get(CAP_PROP_FRAME_HEIGHT)
        fps: int = capture.get(CAP_PROP_FPS)
    
        print("Image frame width: ", int(frame_width))
        print("Image frame height: ", int(frame_height))
        print("Frame rate: ", int(fps))

    joints: list = joints_dict()[pose]['skeleton']
    
    while capture.isOpened():
    
        # Lê o frame da vídeo
        ret, frame = capture.read()
    
        # Se um frame da vídeo não for capturado, termine o vídeo
        if not ret:
            break

        points = model.predict(frame)  # Array numpy com as predições das juntas para cada indivíduo detectado.
        for point in points:
            frame = draw_points_and_skeleton(frame, point, joints)
            
        # Aperta a tela Esc para finalizar o loop
        if waitKey(1) == 27:
            break
        
        #Mostrando cada frame na janela em formato de vídeo.
        imshow('Video Panoptic', resize(frame, (WIDTH, HEIGHT)))
    
    # Libera a captura do vídeo e fecha a janela de visualização
    capture.release()
    destroyAllWindows()
    print('Vídeo Encerrado!')


if __name__ == "__main__":
    #test_single_image()   # Código teste
    main()                 # Código principal
