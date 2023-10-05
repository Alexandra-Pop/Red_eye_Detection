#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <random>

using namespace cv;

//FUNCTIE DE VERIFICARE IS INSIDE IMAGE -> FOLOSITA IN FUNCTIA TESTNEIGHISBACKGROUNG -> FOLOSITA IN CALCUL PERIMETRU:
bool isInside(Mat img, int i, int j) {
	int height = img.rows;
	int width = img.cols;

	if (i > 0 && j > 0 && i < height && j < width) {
		return 1;
	}
	else {
		return 0;
	}
}

//FUNCTIE DE CALCUL AL ARIEI PE O IMAGINE BINARA:
int testComputeArea(Mat img) {
	int area = 0;
	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {
				area++;
			}
		}
	}
	return area;
}

//FUNCTIE DE CALCULARE CENTRU DE MASA:
int* testComputeC(Mat img) {
	int height = img.rows;
	int width = img.cols;

	int sumR = 0;
	int sumC = 0;
	float cR = 0.0f;
	float cC = 0.0f;

	int* center = (int*)calloc(2, sizeof(int));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixel = img.at<uchar>(i, j);
			if (pixel == 0) { //if it is black, si foreground.(pt. ca asa am ales in functia de etichetare).
				sumR += i;
				sumC += j;
			}
		}
	}

	int area = testComputeArea(img);
	cR = (1 / (float)area) * sumR;
	cC = (1 / (float)area) * sumC;

	center[0] = cR;
	center[1] = cC;
	return center;
}

//FUNCTIE CE VERIFICA DACA EXISTA VREUN VECIN AL PIXELULUI (I, J) CARE ESTE BACKGROUND:
bool testNeighbourIsBackground(Mat img, int i, int j) {
	uchar neigh1 = 0, neigh2 = 0, neigh3 = 0, neigh4 = 0, neigh5 = 0, neigh6 = 0, neigh7 = 0, neigh8 = 0;

	if (isInside(img, i - 1, j)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh1 = img.at<uchar>(i - 1, j);
	}
	if (isInside(img, i - 1, j - 1)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh2 = img.at<uchar>(i - 1, j - 1);
	}
	if (isInside(img, i - 1, j + 1)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh3 = img.at<uchar>(i - 1, j + 1);
	}
	if (isInside(img, i + 1, j)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh4 = img.at<uchar>(i + 1, j);
	}
	if (isInside(img, i - 1, j)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh1 = img.at<uchar>(i - 1, j);
	}
	if (isInside(img, i + 1, j + 1)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh5 = img.at<uchar>(i + 1, j + 1);
	}
	if (isInside(img, i + 1, j - 1)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh6 = img.at<uchar>(i + 1, j - 1);
	}
	if (isInside(img, i, j + 1)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh7 = img.at<uchar>(i, j + 1);
	}
	if (isInside(img, i, j - 1)) {
		printf("img = %d\n", img.at<uchar>(i - 1, j));
		neigh8 = img.at<uchar>(i, j - 1);
	}

	if ((neigh1 == 255) || (neigh2 == 255) || (neigh3 == 255) || (neigh4 == 255) || (neigh5 == 255) || (neigh6 == 255) ||
		(neigh7 == 255) || (neigh8 == 255)) {
		return true;
	}
	return false;
}

//FUNCTIE DE CALCULARE PERIMETRU CU VECINATATE DE 8:
int testComputePerimeter(Mat img) {
	int height = img.rows;
	int width = img.cols;
	int p = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {
				if (testNeighbourIsBackground(img, i, j)) {
					p++;
				}
			}
		}
	}
	return p;
}

//FUNCTIE CE TRANSFORMA O IMAGINE RGB INTR-O IMAGINE BINARA -> FOLOSITA LA PRELUAREA UNUI SINGUR OBIECT DINTR-O IMAGINE LABELED:
Mat testRGBtoBWObjectThreshold(Mat img, Vec3b color) {
	int height = img.rows;
	int width = img.cols;
	Mat newImg(height, width, CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixelColor = img.at<Vec3b>(i, j);
			if (pixelColor == color) {
				newImg.at<uchar>(i, j) = 0;
			}
			else {
				newImg.at<uchar>(i, j) = 255;
			}
		}
	}
	return newImg;
}

//FUNCTIE CE INLOCUIESTE FIECARE LABEL NUMBERIC DINTR-O IMAGINE CU O CULOARE DISTINCTA SI RETURNEAZA IMAGINEA LABELED CONTINAND ACELE CULORI:
Mat ColoredLabeledImage(Mat labels) {
	default_random_engine gen;
	uniform_int_distribution<int> d(0, 255);

	int height = labels.rows;
	int width = labels.cols;
	Mat finalSrc = Mat(height, width, CV_8UC3);

	int* colorsR = (int*)calloc(256, sizeof(int));
	int* colorsG = (int*)calloc(256, sizeof(int));
	int* colorsB = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int index = (int)labels.at<uchar>(i, j);
			if ((labels.at<uchar>(i, j) == 0) && (colorsR[index] == 0) && (colorsG[index] == 0) && (colorsB[index] == 0)) {
				colorsR[index] = 255;
				colorsG[index] = 255;
				colorsB[index] = 255;
			}
			else if ((colorsR[index] == 0) && (colorsG[index] == 0) && (colorsB[index] == 0)) {
				uchar r = d(gen);
				uchar g = d(gen);
				uchar b = d(gen);
				colorsR[index] = r;
				colorsG[index] = g;
				colorsB[index] = b;
			}
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int index = (int)labels.at<uchar>(i, j);
			finalSrc.at<Vec3b>(i, j) = Vec3b(colorsB[index], colorsG[index], colorsR[index]);
		}
	}

	free(colorsR);
	free(colorsG);
	free(colorsB);
	return finalSrc;
}

//ARRAY-URI PENTRU REPREZENTAREA VECINATATII DE 4 SI DE 8:
int di4[8] = { 0, 1, 0, -1 };
int dj4[8] = { 1, 0, -1, 0 };

int di8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
int dj8[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

//FUNCTIE CE FACE LABELING PE O IMAGINE APLICAND UN BFS PE IMAGINE:
Mat BFSLabels(Mat src, int flag) {
	int height = src.rows;
	int width = src.cols;
	int label = 0;

	//INITIALIZARE MATRICE FINALA CU 0-URI:
	Mat finalSrc = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			finalSrc.at<uchar>(i, j) = 0; //TOTUL E NEETICHETAT;
		}
	}

	//ALEGEM VECINATATEA IN FUNCTIE DE FLAG:
	int dim = 0;
	int* di, * dj;
	if (flag == 0) {
		dim = 4;
		di = di4;
		dj = dj4;
	}
	else if (flag == 1) {
		dim = 8;
		di = di8;
		dj = dj8;
	}
	else {
		exit(-1);
	}

	//BFS:
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((src.at<uchar>(i, j) == 0) && (finalSrc.at<uchar>(i, j) == 0)) {
				label++;
				std::queue<Point> Q;
				finalSrc.at<uchar>(i, j) = label;
				Q.push({ i, j });
				while (!Q.empty()) {
					Point q = Q.front();
					Q.pop();
					for (int k = 0; k < dim; k++) {
						if (((q.x + di[k]) >= 0) && ((q.x + di[k]) < height) && ((q.y + dj[k]) >= 0) && ((q.y + dj[k]) < width)
							&& (src.at<uchar>(q.x + di[k], q.y + dj[k]) == 0) && (finalSrc.at<uchar>(q.x + di[k], q.y + dj[k]) == 0)) {
							finalSrc.at<uchar>(q.x + di[k], q.y + dj[k]) = label;
							Q.push({ q.x + di[k], q.y + dj[k] });
						}
					}
				}
			}
		}
	}
	return ColoredLabeledImage(finalSrc);
}

//FUNCTIE CE TRANSFORMA DINTR-O IMAGINE GRAYSCALE INTR-O IMAGINE BINARA:
Mat GStoBW(Mat imgGS, int threshold) {
	int height = imgGS.rows;
	int width = imgGS.cols;
	Mat imgBW(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixel = imgGS.at<uchar>(i, j);
			if (pixel >= threshold) {
				imgBW.at<uchar>(i, j) = 0; //CE E MAI DESCHIS LA CULOARE IN IMAGINEA GS VA FI NEGRU IN IMAGINEA BINARY;
			}
			else {
				imgBW.at<uchar>(i, j) = 255; //CE E MAI INCHIS LA CULOARE IN IMAGINEA GS VA FI ALB IN IMAGINEA BINARY;
			}
		}
	}
	return imgBW;
}

//ARRAY-URI CE REPREZINTA KERNEL-URILE DE DIMENSIUNI 3, 5 SI 7:
int di3[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
int dj3[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

int di5[] = { -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
int dj5[] = { -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2 };

int di7[] = { -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 };
int dj7[] = { -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3 };

//FUNCTIE CE REALIZEAZA INCHIDEREA PE O IMAGINE, CU DIMENSIUNEA KERNELULUI DE 3, 5 SAU 7:
Mat ClosingOperation(Mat src, int dim) {
	int height = src.rows;
	int width = src.cols;
	Mat finalSrc = src.clone();
	int* di, * dj;

	if (dim == 3) {
		di = di3;
		dj = dj3;
	}
	else if (dim == 5) {
		di = di5;
		dj = dj5;
	}
	else if (dim == 7) {
		di = di7;
		dj = dj7;
	}
	else {
		exit(-1);
	}

	//INCHIDERE:
	//DILATARE:
	for (int i = dim - 2; i < height - (dim / 2); i++) {
		for (int j = dim - 2; j < width - (dim / 2); j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < dim * dim; k++) {
					if (src.at<uchar>(i + di[k], j + dj[k]) != 0) {
						finalSrc.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}
	}
	Mat newSrc = finalSrc.clone();
	//EROZIUNE:
	for (int i = dim - 2; i < height - (dim / 2); i++) {
		for (int j = dim - 2; j < width - (dim / 2); j++) {
			if (newSrc.at<uchar>(i, j) == 0) {
				for (int k = 0; k < dim * dim; k++) {
					if (newSrc.at<uchar>(i + di[k], j + dj[k]) != 0) {
						finalSrc.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}
	return finalSrc;
}

//FUNCTIE CE REALIZEAZA DESCHIDEREA PE O IMAGINE CU UN KERNEL DE DIMENSIUNE 3, 5 SAU 7:
Mat OpeningOperation(Mat src, int dim) {
	int height = src.rows;
	int width = src.cols;
	Mat finalSrc = src.clone();
	int* di, * dj;

	if (dim == 3) {
		di = di3;
		dj = dj3;
	}
	else if (dim == 5) {
		di = di5;
		dj = dj5;
	}
	else if (dim == 7) {
		di = di7;
		dj = dj7;
	}
	else {
		exit(-1);
	}

	//DESCHIDERE:
	//EROZIUNE:
	for (int i = dim - 2; i < height - (dim / 2); i++) {
		for (int j = dim - 2; j < width - (dim / 2); j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < dim * dim; k++) {
					if (src.at<uchar>(i + di[k], j + dj[k]) != 0) {
						finalSrc.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}
	Mat newSrc = finalSrc.clone();
	//DILATARE:
	for (int i = dim - 2; i < height - (dim / 2); i++) {
		for (int j = dim - 2; j < width - (dim / 2); j++) {
			if (newSrc.at<uchar>(i, j) == 0) {
				for (int k = 0; k < dim * dim; k++) {
					if (newSrc.at<uchar>(i + di[k], j + dj[k]) != 0) {
						finalSrc.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}
	}
	return finalSrc;
}

//FUNCTIE CE FACE TRANSFORMAREA DIN SPATIUL DE CULOARE RGB IN SPATIUL DE CULOARE CIELab:
Mat RGBToLab(Mat src, int height, int width) {
	Mat XYZSrc = Mat(height, width, CV_32FC3);
	Mat LabSrc = Mat(height, width, CV_32FC3);
	//CONVERT FROM BGR TO XYZ AND FROM XYZ TO CIELab COLOR SPACE:
	float rgbToxyz[3][3] =
	{
		{ 0.4124, 0.3576, 0.1805 },
		{ 0.2126, 0.7152, 0.0722 },
		{ 0.0193, 0.1192, 0.9505 }
	};
	float X = 0.0, Y = 0.0, Z = 0.0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float pixel0 = src.at<Vec3b>(i, j)[0];
			pixel0 /= 255.0;
			if (pixel0 > 0.04045) {
				pixel0 = pow((pixel0 + 0.055) / 1.055, 2.4);
			}
			else {
				pixel0 = pixel0 / 12.92;
			}
			pixel0 *= 100;

			float pixel1 = src.at<Vec3b>(i, j)[1];
			pixel1 /= 255.0;
			if (pixel1 > 0.04045) {
				pixel1 = pow((pixel1 + 0.055) / 1.055, 2.4);
			}
			else {
				pixel1 = pixel1 / 12.92;
			}
			pixel1 *= 100;

			float pixel2 = src.at<Vec3b>(i, j)[2];
			pixel2 /= 255.0;
			if (pixel2 > 0.04045) {
				pixel2 = pow((pixel2 + 0.055) / 1.055, 2.4);
			}
			else {
				pixel2 = pixel2 / 12.92;
			}
			pixel2 *= 100;

			XYZSrc.at<Vec3f>(i, j)[0] = rgbToxyz[0][0] * pixel2 + rgbToxyz[0][1] * pixel1 + rgbToxyz[0][2] * pixel0; //X
			X = XYZSrc.at<Vec3f>(i, j)[0];
			X /= 95.047;
			if (X > 0.008856) {
				X = pow(X, 1.0 / 3);
			}
			else {
				X = (7.787 * X) + (16.0 / 116);
			}

			XYZSrc.at<Vec3f>(i, j)[1] = rgbToxyz[1][0] * pixel2 + rgbToxyz[1][1] * pixel1 + rgbToxyz[1][2] * pixel0; //Y
			Y = XYZSrc.at<Vec3f>(i, j)[1];
			Y /= 100.0;
			if (Y > 0.008856) {
				Y = pow(Y, 1.0 / 3);
			}
			else {
				Y = (7.787 * Y) + (16.0 / 116);
			}

			XYZSrc.at<Vec3f>(i, j)[2] = rgbToxyz[2][0] * pixel2 + rgbToxyz[2][1] * pixel1 + rgbToxyz[2][2] * pixel0; //Z
			Z = XYZSrc.at<Vec3f>(i, j)[2];
			Z /= 108.883;
			if (Z > 0.008856) {
				Z = pow(Z, 1.0 / 3);
			}
			else {
				Z = (7.787 * Z) + (16.0 / 116);
			}

			LabSrc.at<Vec3f>(i, j)[0] = (116 * Y) - 16; //L
			LabSrc.at<Vec3f>(i, j)[1] = 500 * (X - Y); //a
			LabSrc.at<Vec3f>(i, j)[2] = 200 * (Y - Z); //b
		}
	}
	return LabSrc;
}

//FUNCTIE CE CALCULEAZA, PENTRU FIECARE PIXEL IN SPATIUL DE CULOARE CIELab, DISTANTA EUCLIDIANA INTRE CANALELE SALE DE CULOARE a si b, SI CANALELE a si b
//ALE UNUI PRESUPUS PIXEL IDEAL DE CULOARE ROSIE, APOI REALIZEAZA O IMAGINE GRAYSCALE, FACAND O SCALARE IN INTERVALUL [0, 255], PIXELII CU VALORI MARI DE D
//FIIN MAI APROPIATI DE 0 (NEGRU), IAR PIXELII CU VALORI MICI ALE LUI D FIIND MAI APROPIATI DE 255 (ALB):
Mat rednessMask(Mat LabSrc, int height, int width) {
	float typicalRed[] = { 31.387, 54.475, 32.72 };
	float* dArray = (float*)calloc(height * width, sizeof(float));
	float min = FLT_MAX, max = FLT_MIN;

	//COMPUTE THE DISTANCE BETWEEN THE CURRENT COLOR AND THE IDEAL RED COLOR, COMPUTE THE MAX AND THE MIN OF THIS VALUE AND STORE IT IN AN ARRAY:
	Mat mask = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//distanta euclidiana intre cele 2 culori dpdv al a si b:
			float d = pow(pow((LabSrc.at<Vec3f>(i, j)[1] - typicalRed[1]), 2) + pow((LabSrc.at<Vec3f>(i, j)[2] - typicalRed[2]), 2), 0.5);
			dArray[i * width + j] = d;
			if (d > max) {
				max = d;
			}
			if (d < min) {
				min = d;
			}
		}
	}

	//COMPUTE THE MASK:
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			mask.at<uchar>(i, j) = 255 * ((max - dArray[i * width + j]) / (max - min));
		}
	}
	return mask;
}

//FUNCTIE CARE SCOATE UN OBIECT CU UN ANUMIT LABEL DINTR-O IMAGINE RGB:
Mat removeLabeledObject(Mat src, int height, int width, Vec3b label) {
	Mat finalSrc = src.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<Vec3b>(i, j) == label) {
				finalSrc.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
	}
	return finalSrc;
}

//FUNCTIE CARE ADAUGA UN PIXELII OBIECT AI UNEI IMAGINI BINARE IN ALTA IMAGINE BINARA -> UTILIZAT PENTRU RETINEREA COMPONENTELOR GASITE CARE RESPECTA
//CERINTELE DATE:
Mat addObject(Mat filterSrc, Mat src, int height, int width) {
	Mat finalSrc = filterSrc.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				finalSrc.at<uchar>(i, j) = src.at<uchar>(i, j);
			}
		}
	}
	return finalSrc;
}

//FUNCTIE CARE CALCULEAZA DISTANTA MAXIMA DE LA CENTRUL DE MASA AL UNUI OBIECT LA RESTUL PIXELILOR SAI:
int computeRadius(Mat src, int height, int width, int xCenter, int yCenter) {
	float distance = 0.0;
	float max = FLT_MIN;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				distance = pow(pow((i - xCenter), 2) + pow((j - yCenter), 2), 0.5);
				if (distance > max) {
					max = distance;
				}
			}
		}
	}
	return max;
}

//FUNCTIE CARE GASESTE MIN SI MAX ALE COORDONATELOR X SI Y ALE UNUI OBIECT DINTR-O IMAGINE BINARA:
vector<int> findMinAndMax(Mat src, int height, int width) {
	int maxX = INT_MIN, maxY = INT_MIN;
	int minX = INT_MAX, minY = INT_MAX;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				if (i < minX) { minX = i; }
				if (i > maxX) { maxX = i; }
				if (j < minY) { minY = j; }
				if (j > maxY) { maxY = j; }
			}
		}
	}
	vector<int> values;
	values.push_back(minX); values.push_back(maxX); values.push_back(minY); values.push_back(maxY);
	return values;
}

//FUNCTIA PRINCIPALA DE PROIECT:
void proiect() {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat src = imread(fname, IMREAD_COLOR);
	int height = src.rows;
	int width = src.cols;

	//INITIAL, REALIZEZ CONVERSIA DIN SPATIUL RGB IN SPATIUL CIELab:
	Mat LabSrc = RGBToLab(src, height, width);

	//APOI, AVAND IMAGINEA CONVERTITA, II CALCULEZ MASCA PENTRU CULORI ROSII:
	Mat mask = rednessMask(LabSrc, height, width);

	//MASCA O TRANSFORM APOI INTR-O IMAGINE BINARA, LASAND SA TREACA DOAR VALORILE CARE AU O INTENSITATE > 175 (PIXELII CONSIDERATI ROSII):
	Mat binarySrc = GStoBW(mask, 175); //GS to Binary -> Threshold = 175;

	//REALIZEZ O OPERATIE DE INCHIDERE UN UN KERNEL DE DIMENSIUNE 7 PE IMAGINEA BINARA REZULTATA:
	Mat closedSrc = ClosingOperation(binarySrc, 7); //REZULTATELE CELE MAI BUNE;

	//FAC LABELING PE OBIECTELE CONSIDERATE ROSII, CU VECINATATE DE 8:
	Mat labeledSrc = BFSLabels(closedSrc, 1);

	//PREIAU FIECARE OBIECT LABELED, II CALCULEZ CENTRUL DE GREUTATE, RAZA MAXIMA SI DESENEZ UN CERC CU CENTRUL SI RAZA SA, IL UMPLU, CALCULEZ ARIA
	//CERCULUI, ARIA OBIECTULUI MEU, SI VERIFIC DACA DIFERENTA DINTRE ELE E MAI MICA DECAT O EROARE:
	Mat interSrc = labeledSrc.clone();
	Mat interBinarySrc = Mat(height, width, CV_8UC1);
	Mat filteredSrc = Mat(height, width, CV_8UC1, Scalar(255, 255, 255));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (interSrc.at<Vec3b>(i, j) != Vec3b(255, 255, 255)) {
				//PREIAU OBIECTUL CU UN LABEL CARE ESTE DIFERIT DE ALB:
				interBinarySrc = testRGBtoBWObjectThreshold(interSrc, interSrc.at<Vec3b>(i, j));
				//IL SCOT DIN IMAGINEA LABELED PENTRU A NU-L PRELUCRA DE MAI MULTE ORI:
				interSrc = removeLabeledObject(interSrc, height, width, interSrc.at<Vec3b>(i, j));

				//INITIALIZEZ IMAGINEA CE VA CONTINE CERCUL INVELITOR ACELUI OBIECT LA CULOAREA ALB:
				Mat circleSrc = Mat(height, width, CV_8UC1, Scalar(255, 255, 255));

				//CALCULEZ CENTRUL SI IL DESENEZ IN IMAGINEA CE VA CONTINE CERCUL:
				int* center = testComputeC(interBinarySrc);
				Mat centerSrc = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
				centerSrc.at<Vec3b>(center[0], center[1]) = (0, 0, 255);

				//CALCULEZ RAZA SI APOI DESENEZ CERCUL CU FUNCTIA PREDEFINITA CIRCLE:
				float radius = computeRadius(interBinarySrc, height, width, center[0], center[1]);
				circle(circleSrc, Point(center[1], center[0]), radius, Scalar(0, 0, 0), FILLED);
				printf("radius = %f\n", radius);

				//CALCULEZ ARIA CERCULUI SI A OBIECTULUI:
				int objectArea = testComputeArea(interBinarySrc);
				int circleArea = testComputeArea(circleSrc);
				printf("Object Area = %d; Circle Area = %d\n", objectArea, circleArea);

				//CALCULEZ COORDONATELE DE MIN SI MAX ALE OBIECTULUI CURENT:
				vector<int> minMax = findMinAndMax(interBinarySrc, height, width);
				printf("Diferenta arii = %f\n", (abs(circleArea - objectArea) / (float)circleArea));

				//CALCULEZ UN RAPORT INTRE WIDTH SI HEIGHT ALE OBIECTULUI CURENT:
				float ratio = ((float)minMax[3] - (float)minMax[2]) / ((float)minMax[1] - (float)minMax[0]);

				//DACA DIFERENTA IN ARII A OBIECTULUI CURENT SI CERCULUI, NORMALIZATA, ESTE MAI MICA DE 0.4 SI RATIO-UL PE INALTIME SI LATIME ESTE
				//MAI MARE DE 1/2 SI MAI MIC DE 2 ATUNCI PUNCTUL ESTE CONSIDERAT UN CANDIDAT DE OCHI ROSU SI ESTE PUS IN IMAGINEA BINARA FINALA:
				if (((abs(circleArea - objectArea) / (float)circleArea) < 0.4) && (ratio > 0.5) && (ratio < 2.0)) {
					printf("Ales\n");
					filteredSrc = addObject(filteredSrc, interBinarySrc, height, width);
				}
			}
		}
	}

	//INLOCUIRE CULOARE ROSIE IN OBIECTELE DETECTATE CU O ALTA CULOARE (GRI IN FUNCTIE DE INTENSITATE):
	Mat LabSrcCorrected = LabSrc.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (filteredSrc.at<uchar>(i, j) == 0) {
				//DACA IN IMAGINEA BINARA FINALA AVEM UN PUNCT OBIECT, II CALCULAM LUMINOZITATEA ACESTUIA IN IMAGINEA FINALA:
				//CU CAT ARE O CULOARE MAI ROSIATICA, CU ATAT VALOAREA DIN MASCA ESTE MAI APROPIATA DE 255, DECI LUMINOZITATEA DIN
				//IMAGINEA INITIALA VA AVEA O PONDERE MAI MICA IN LUMINOZITATEA FINALA:
				float newL = ((1 - (mask.at<uchar>(i, j) / 255.0)) * LabSrc.at<Vec3f>(i, j)[0]);
				//CULOAREA FINALA CARE VA FI PUSA IN IMAGINEA FINALA IN SPATIUL DE CULOARE CIELab, VA FI CULOAREA NEGRU CU O LUMINOZITATE
				//IN FUNCTIE DE INTENSITATEA CULORII DE ROSU DIN IMAGINEA INITIALA:
				LabSrcCorrected.at<Vec3f>(i, j) = Vec3f(newL, 0.0, 0.0);
			}
		}
	}

	//CONVERTIM INAPOI DIN CIELab IN RGB CU FUNCTIA PREDEFINITA cvtColor:
	Mat correctedSrc = Mat(height, width, CV_8UC3);
	cvtColor(LabSrcCorrected, correctedSrc, COLOR_Lab2BGR);

	//AFISAM IMAGINEA ORIGINALA, MASCA DE ROSU, IMAGINEA BINARA A MASTII, IMAGINEA INCHISA A IMAGINII BINARE, IMAGINEA LABELED A IMAGINII BINARE,
	//IMAGINEA BINARA FINALA CU OBIECTELE CONSIDERATE OCHI ROSII, SI IMAGINEA FINALA CU ACEI PIXELI CORECTATI:
	imshow("original image", src);
	imshow("redmask image", mask);
	imshow("binary image", binarySrc);
	imshow("closed image", closedSrc);
	imshow("labeled image", labeledSrc);
	imshow("filtered image", filteredSrc);
	imshow("corrected image", correctedSrc);
	waitKey(0);
}

void fourierTransform() {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat src = imread(fname, IMREAD_GRAYSCALE);
	int height = src.rows;
	int width = src.cols;

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//Centering: daca i+j este par => ramane la fel, daca e impar => = - srcf.
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			srcf.at<float>(i, j) = ((i + j) & 1) ? -srcf.at<float>(i, j) : srcf.at<float>(i, j);
		}
	}

	//Tranformata Fourier cu metoda predefinita din OpenCV:
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	Mat finalLogMag = mag.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			mag.at<float>(i, j) = 1 + mag.at<float>(i, j);
		}
	}
	log(mag, finalLogMag);

	imshow("original image", src);
	imshow("fourier image", channels[0]);
	imshow("fourier image 2", channels[1]);
	imshow("log mag image", finalLogMag);
	waitKey(0);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Proiect\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		int returnedValue = scanf("%d", &op);
		switch (op)
		{
		case 1:
			proiect();
			break;
		case 2:
			fourierTransform();
			break;
		}
	} while (op != 0);
	return 0;
}