import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import chess.svg
import cairosvg
import chess
import cv2

class Helpers:
    @staticmethod
    def Board2Fen(board:np.ndarray)->str:
        """
        Arguments:
        board:np.ndarray
        Converts np.ndarray to FEN format (str).
        example input format:
        board_fen= [["br","bn","bb","bq","bk","bb","bn","br"],
                    ["bp","bp","bp","bp","bp","bp","bp","bp"],
                    ["--","--","--","--","--","--","--","--"],
                    ["--","--","--","--","--","--","--","--"],
                    ["--","--","--","--","--","--","--","--"],
                    ["--","--","--","--","--","--","--","--"],
                    ["wp","wp","wp","wp","wp","wp","wp","wp"],
                    ["wr","wn","wb","wq","wk","wb","wn","wr"]]
        Returns:
        Fen format string.
        """
        mystr=""
        counter=1
        index=0
        for row in board:
            for i in range(8):
                if(row[i][0]=="b"):
                    mystr+=row[i][1].lower()
                if(row[i][0]=="w"):
                    mystr+=row[i][1].capitalize()
                if(i<7):
                    if(row[i]=="--" and row[i+1]=="--" ):
                        counter+=1
                    if(row[i]=="--" and row[i+1]!="--"):
                        mystr+=str(counter)
                        counter=1
                elif(row[7]=="--"):
                    mystr+=str(counter)
            if(index!=7):        
                    mystr+="/"          
            counter=1 
            index+=1
        return mystr
    
    @staticmethod
    def ChessBoardSaveAsPNG(chessboard:np.ndarray)->None:
        """
        Takes chessboard as np.ndarray and save chessboard in png format.

        Arguments:
        chessboard:np.ndarray
        """
        fen=Helpers.Board2Fen(chessboard)
        print("Fen:",fen)
        board=chess.Board(fen)
        board_svg=chess.svg.board(board, size=600, coordinates=False)
        img_png = cairosvg.svg2png(board_svg)
        img = Image.open(BytesIO(img_png))
        plt.imshow(img)
        plt.rc("font",size=8)
        plt.axis('off')
        plt.title(f"Fen:{fen}",loc="center")
        plt.savefig("outputs/chessboard.png")
        plt.show()

    @staticmethod
    def FindCorrespondingSquare(centers:np.ndarray,point:np.ndarray,piece:list)->np.ndarray:
        """
        Arguments:
        centers: chessboard square center coordinates
        point: detected pieces cordinates
        piece: detected pieces classes

        Returns:
        Chessboard as np.ndarray with detected pieces added on it while blank 
        squares remains "--".
        """
        centers=np.array(centers)
        point=np.array(point)
        chessboard=np.full((8,8),"--")
        for i,p in enumerate(point):
            x=(p[0]+p[2])/2
            y=p[1]+(abs(p[1]-p[3])*0.75)
            res=np.square(np.subtract(centers,[x,y]).__abs__()).sum(axis=1)
            index=np.sqrt(res).argmin()
            t,k=int(index/8),index%8
            chessboard[t][k]=piece[i]
        return chessboard
    
    @staticmethod
    def FindSquareCenters(corners)->np.ndarray:
        """
        Calculates the squares centers coordinates with given corner points.
        """
        square_centers=[]
        c1,c2=0,10
        for i in range(8):
            for j in range(8):
                x_diff=abs(corners[c1][0]-corners[c2][0])/2
                y_diff=abs(corners[c1][1]-corners[c2][1])/2
                x,y=int(round(corners[c1][0]+x_diff)),int(round(corners[c1][1]+y_diff))
                square_centers.append([x,y])
                c1+=1
                c2+=1
            c1+=1
            c2+=1
        return square_centers

    @staticmethod
    def GetChessboardSquareCorners(corners)->np.ndarray:
        """
        Calculates the squares corners coordinates with given chessboard contour
        corner points.
        """
        base_size=112
        width,height=base_size*8,base_size*8
        imgTl = [0,0]
        imgTr = [width,0]
        imgBr = [width,height]
        imgBl = [0,height]
        img_params = np.float32([imgTl,imgTr,imgBr,imgBl])
        corners=np.float32([corners[0],corners[1],corners[2],corners[3]])

        matrix = cv2.getPerspectiveTransform(corners,img_params)
        coef1,coef2=0,0
        actual_corners=[]
        for i in range(9):
            for j in range(9):
                res=np.matmul(np.linalg.inv(matrix),[[coef1],[coef2],[1]])
                x,y=int(round((res[0]/res[2])[0])),int(round((res[1]/res[2])[0]))
                actual_corners.append([x,y])
                coef1+=base_size
            coef1=0
            coef2+=base_size
        return actual_corners
    
    @staticmethod
    def RelocateCorners(corners):
        """
        Relocates the detected chessboard contour corner points with top_left,
        top_right,bottom_right,bottom_left order.
        """
        c=[list(c) for c in corners]
        top_left=list(corners[np.sum(corners,axis=1).argmin()])
        bottom_right=list(corners[np.sum(corners,axis=1).argmax()])
        c.pop(c.index(top_left))
        c.pop(c.index(bottom_right))
        bottom_left= c[1] if c[0][0]>c[1][0] else c[0]
        top_right= c[1] if c[0][0]<c[1][0] else c[0]
        return np.array([top_left,top_right,bottom_right,bottom_left])