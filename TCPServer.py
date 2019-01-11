import socketserver
import cv2
import numpy as np
from termcolor import colored
import urllib.request

import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as T
from PIL import Image

## CONST VALUE
HELLO = 1
IMGTRANSFER = 2
READ_BYTES = 16384

PATH_MODEL = {}
for i in range(1, 26):
    PATH_MODEL[i] = ("./models/model{}.ckpt".format(i))


def asciiArt():
    print("")
    print("          _____                    _____                    _____                    _____          ")
    print("         /\    \                  /\    \                  /\    \                  /\    \         ")
    print("        /::\    \                /::\____\                /::\    \                /::\    \        ")
    print("       /::::\    \              /:::/    /                \:::\    \              /::::\    \       ")
    print("      /::::::\    \            /:::/   _/___               \:::\    \            /::::::\    \      ")
    print("     /:::/\:::\    \          /:::/   /\    \               \:::\    \          /:::/\:::\    \     ")
    print("    /:::/__\:::\    \        /:::/   /::\____\               \:::\    \        /:::/  \:::\    \    ")
    print("    \:::\   \:::\    \      /:::/   /:::/    /               /::::\    \      /:::/    \:::\    \   ")
    print("  ___\:::\   \:::\    \    /:::/   /:::/   _/___    ____    /::::::\    \    /:::/    / \:::\    \  ")
    print(" /\   \:::\   \:::\    \  /:::/___/:::/   /\    \  /\   \  /:::/\:::\    \  /:::/    /   \:::\ ___\ ")
    print("/::\   \:::\   \:::\____\|:::|   /:::/   /::\____\/::\   \/:::/  \:::\____\/:::/____/     \:::|    |")
    print("\:::\   \:::\   \::/    /|:::|__/:::/   /:::/    /\:::\  /:::/    \::/    /\:::\    \     /:::|____|")
    print(" \:::\   \:::\   \/____/  \:::\/:::/   /:::/    /  \:::\/:::/    / \/____/  \:::\    \   /:::/    / ")
    print("  \:::\   \:::\    \       \::::::/   /:::/    /    \::::::/    /            \:::\    \ /:::/    /  ")
    print("   \:::\   \:::\____\       \::::/___/:::/    /      \::::/____/              \:::\    /:::/    /   ")
    print("    \:::\  /:::/    /        \:::\__/:::/    /        \:::\    \               \:::\  /:::/    /    ")
    print("     \:::\/:::/    /          \::::::::/    /          \:::\    \               \:::\/:::/    /     ")
    print("      \::::::/    /            \::::::/    /            \:::\    \               \::::::/    /      ")
    print("       \::::/    /              \::::/    /              \:::\____\               \::::/    /       ")
    print("        \::/    /                \::/____/                \::/    /                \::/____/        ")
    print("         \/____/                  ~~                       \/____/                  ~~              ")
    print()
    



def image_transform(image): # imageSize is Tuple
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5))
    ])

    return transform(image)

class TCPHandler(socketserver.BaseRequestHandler):
    packet = bytearray()
    totalbytes = 0

    def handle(self):
        print("")
        print('Client is connected: {0}'.format(self.client_address[0]))
        close = 0

        #while not close:
        sock = self.request

        result = bytearray()
        images = []

        packet = self.packet + sock.recv(READ_BYTES)
        #print(len(packet))
        protocol = int.from_bytes(packet[0:4], byteorder="little")
        packet = packet[4:]

        if protocol == HELLO:  # Hello

            # 전송되는 이미지 수
            numberOfImg = int.from_bytes(packet[0:4], byteorder="little")
            print("이미지 수 : %d" % numberOfImg)
            # 전송되는 전체 바이트 수
            totalbytes = int.from_bytes(packet[4:8], byteorder="little")

            packet = packet[8:]
            sock.send("OK".encode())
            numberOfRecvImg = 0
            while numberOfImg > numberOfRecvImg:
                packet = self.packet + sock.recv(READ_BYTES)
                #print(len(packet))
                protocol = int.from_bytes(packet[0:4], byteorder="little")
                packet = packet[4:]

                if protocol == IMGTRANSFER:  # ImageTransfer
                    if len(packet) < 8:
                        packet = packet + sock.recv(READ_BYTES)
                    # if totalbytes < READ_BYTES :
                    #    READ_BYTES = totalbytes

                    filenamelen = int.from_bytes(packet[0:4], byteorder="little")
                    imgsizelen = int.from_bytes(packet[4:8], byteorder="little")
                    packet = packet[8:]

                    if len(packet) < filenamelen + imgsizelen:
                        packet = packet + sock.recv(READ_BYTES)

                    filename = packet[:filenamelen].decode()
                    img = packet[filenamelen:imgsizelen]

                    packet = packet[imgsizelen + filenamelen:]

                    img = np.frombuffer(img, dtype=np.uint8)
                    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

                    images.append(img)
                    numberOfRecvImg += 1
                    sock.send("OK".encode())

        packet = sock.recv(1024)
        # 받은 이미지 수와 처음 전송된 이미지의 수가 같으면 ==> 정상
        if numberOfRecvImg == numberOfImg :
            result.append(numberOfRecvImg)
        else:
            # ERROR occured
            print("ERROR")
        print("numberOfRecvImg : %d" % numberOfRecvImg)
        model.eval()
        for i in range(numberOfRecvImg):
           cv2.imwrite("img" + str(i) + ".jpg", images[i])
        

        print("\nStart Inspection")
        with torch.no_grad():
            for i in range(numberOfRecvImg):
                # Write image classification code here
                isCrack = 0
                if i > 24:
                    break
                
                gsImage = Image.fromarray(images[i])
                gsImage = image_transform(gsImage)
                gsImage.to(device)
                gsImage.unsqueeze_(0)
                out = modelList[i+1](gsImage)
                
                if torch.max(out.data, 1) == 0:
                    isCrack = 1

                else:
                    isCrack = 0
                # append result of each image in this way
                # 1 : CRACK
                # 0 : NORMAL
                result.append(isCrack)

        print("End Inspection")

        # Send result
        sock.send(result)

modelList = {}
if __name__ == '__main__':
    bindIP = urllib.request.urlopen('http://ifconfig.me/ip').read()
    bindPort = 8888
    asciiArt()
    print("Server External IP : " + str(bindIP.decode()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(1, 26):
        modelList[i] = resnet18()
        modelList[i] = modelList[i].to(device)
        if torch.cuda.device_count() > 1:
            modelList[i] = torch.nn.DataParallel(modelList[i])
        modelList[i].load_state_dict(torch.load(PATH_MODEL[i]))

    server = socketserver.ThreadingTCPServer((bindIP, bindPort), TCPHandler)
    print('Server START')
    server.serve_forever()
