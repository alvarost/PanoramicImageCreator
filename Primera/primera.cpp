#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include <cmath>
#include <exception>
#include <time.h>

static const std::string OPENCV_WINDOW = "Image window";

namespace enc = sensor_msgs::image_encodings;
using namespace cv;

class PanoramCreator {
public:
	enum SOURCE {bag, camera, kinect};
	enum DETECTOR {SIFT_D, SURF_D, FAST, MSER, ORB_D};
	enum EXTRACTOR {SIFT_E, SURF_E};

	PanoramCreator(SOURCE src, DETECTOR dtc, EXTRACTOR ext, float filterValue1, float filterValue2 ) : it_(nh_) {
		this->first = true;
		this->filter1 = filterValue1;
		this->filter2 = filterValue2;
		this->detector = dtc;
		this->extractor = ext;
		this->numberGoodMatches = 0;
		this->numberKeypoints = 0;
		this->numberMatches = 0;
		this->numberOfProcessedFrames = 0;

		switch(src){
		case bag:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCb, this);
			break;
		case camera:
			image_sub_ = it_.subscribe("/camera/image_raw", 1,
			&PanoramCreator::imageCb, this);
			break;
		case kinect:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCb, this);
			break;
		default:
			image_sub_ = it_.subscribe("/camera/rgb/image_color", 1,
			&PanoramCreator::imageCb, this);
		}

	}

	~PanoramCreator() {
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg) {
		this->numberOfProcessedFrames++;
		cv_bridge::CvImagePtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8); //convirtiendo la imagen a CV
		} catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		vector<KeyPoint> keyPoints;

		Mat imageColor;
		Mat src_gray;

		if (this->first) { //si es la primera imagen
			cvtColor(cv_ptr->image, this->firstImg, CV_BGR2GRAY); //metemos la imagen en firstImg en gris
			this->first = false;
			this->firstImgColor = cv_ptr->image; //almacenamos la imagen en color
		} else {//si es la segunda
			cvtColor(cv_ptr->image, this->secondImg, CV_BGR2GRAY); //la almacenamos en gris
			this->secondImgColor = cv_ptr->image; //almacenamos la imagen en color
			this->processPanoram(); //las procesamos
		}
	}

	//Método: ProcessPanoram:
	//*******************************************************
	void processPanoram() {
		Mat mask;
		vector<KeyPoint> keyPointsFirst;
		vector<KeyPoint> keyPointsSecond;
		Mat imageColorFirst;
		Mat imageColorSecond;

		cv::FeatureDetector* detector;
		//Cogemos el detector seleccionado
		switch(this->detector){
		case SIFT_D:
			detector = new SiftFeatureDetector();
			break;
		case SURF_D:
			detector = new SurfFeatureDetector();
			break;
		case FAST:
			detector = new FastFeatureDetector();
			break;
		case MSER:
			detector = new MserFeatureDetector();
			break;
		case ORB_D:
			detector = new OrbFeatureDetector();
			break;
		default:
			detector = new OrbFeatureDetector();
		}

		detector->detect(this->firstImg, keyPointsFirst, mask); //almacenamos los keypoints de la primera imagen
		detector->detect(this->secondImg, keyPointsSecond, mask); //almacenamos los keypoints de la segunda imagen
		this->numberKeypoints += keyPointsSecond.size();

		for (size_t i = 0; i < keyPointsFirst.size(); i++) { //Pintamos los keypoints en la primera imagen
			circle(imageColorFirst, keyPointsFirst[i].pt, 3, CV_RGB(255, 0, 0));
		}
		for (size_t i = 0; i < keyPointsSecond.size(); i++) { //Pintamos los keypoints en la segunda imagen
			circle(imageColorSecond, keyPointsSecond[i].pt, 3, CV_RGB(255, 0, 0));
		}

		//imshow("FirstImage", imageColorFirst); //Mostramos la primera imagen
		//imshow("SecondImage", imageColorSecond); //Mostramos la segunda imagen

		cv::DescriptorExtractor* extractor;

		//Elegimos el extractor seleccionado
		switch(this->extractor){
		case SIFT_E:
			extractor = new SiftDescriptorExtractor();
			break;
		case SURF_E:
			extractor = new SurfDescriptorExtractor();
			break;
		default:
			extractor = new OrbDescriptorExtractor();
		}

		Mat desciptorsImg1, desciptorsImg2;

		extractor->compute(this->firstImg, keyPointsFirst, desciptorsImg1); //Obtenemos los descriptores de la primera imagen
		extractor->compute(this->secondImg, keyPointsSecond, desciptorsImg2); //Obtenemos los descriptores de la segunda imagen


		FlannBasedMatcher matcher;
		std::vector< std::vector<DMatch> > matches;
		matcher.knnMatch(desciptorsImg1, desciptorsImg2, matches, 2); // Encontramos 2 más cercanas

		this->numberMatches += matches.size();

		double max_dist = 0; double min_dist = 100, second_min = 100;

		//Calculo de las distancias maxima y minima de los keypoints
		for( int i = 0; i < desciptorsImg1.rows; i++ ){
			double dist = matches[i][0].distance;
				if(dist < min_dist)
					min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}

		std::vector<DMatch> good_matches;
		for(int i=0; i<desciptorsImg1.rows; i++){ // Esto lo hacemos por separado para poder analizar cuantos quitamos con cada uno
			if(matches[i][0].distance < this->filter1*matches[i][1].distance && matches[i][0].distance <
					this->filter2*min_dist)
				good_matches.push_back(matches[i][0]);
		}
		this->numberGoodMatches +=good_matches.size();

		Mat img_matches;
		drawMatches(this->firstImg, keyPointsFirst, this->secondImg, keyPointsSecond,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


		//Mostramos los matches encontrados
		//imshow("Good Matches", img_matches);
		std::vector<Point2f> kImg1;
		std::vector<Point2f> kImg2;

		//Obtenemos los keypoints dados por buenos
		for( int i = 0; i < good_matches.size(); i++ ){
			kImg1.push_back( keyPointsFirst[ good_matches[i].queryIdx ].pt );
			kImg2.push_back( keyPointsSecond[ good_matches[i].trainIdx ].pt );
		}

		//Si tenemos 4 o más matches
		if(good_matches.size() >= 4){
			Mat H = findHomography( kImg1, kImg2, CV_RANSAC ); //Aplicamos RANSAC
			const std::vector<Point2f> points_ant_transformed(keyPointsFirst.size());
			std::vector<Point2f> keypoints_ant_vector(keyPointsFirst.size());
			cv::KeyPoint::convert(keyPointsFirst,keypoints_ant_vector);

			//transformamos los puntos de la imagen anterior
			perspectiveTransform( keypoints_ant_vector, points_ant_transformed, H);

			//creamos una copia de la imagen actual que usaremos para dibujar
			Mat transformed_image;
			cvtColor(this->secondImg, transformed_image, CV_GRAY2BGR);

			//los que esten mas lejos que este parametro se consideran outliers (o que la transformacion está mal calculada)
			float distance_threshold=10.0;
			int contdrawbuenos=0;
			int contdrawmalos=0;


			for ( int i =0;i<good_matches.size();i++){
				int ind        = good_matches.at(i).trainIdx ;
				int ind_Ant    = good_matches.at(i).queryIdx;

				cv::Point2f p=        keyPointsSecond.at(ind).pt;
				cv::Point2f p_ant=    points_ant_transformed[ind_Ant];

				circle( transformed_image, p_ant, 5, Scalar(255,0,0), 2, 8, 0 ); //ant blue
				circle( transformed_image, p, 5, Scalar(0,255,255), 2, 8, 0 ); //current yellow

				Point pointdiff = p - points_ant_transformed[ind_Ant];
					float distance_of_points=cv::sqrt(pointdiff.x*pointdiff.x + pointdiff.y*pointdiff.y);

				if(distance_of_points < distance_threshold){ // los good matches se pintan con un circulo verde mas grand
					contdrawbuenos++;
					circle( transformed_image, p, 9, Scalar(0,255,0), 2, 8, 0 ); //current red
				}else{
					contdrawmalos++;
					line(transformed_image,p,p_ant,Scalar(0, 0, 255),1,CV_AA);
				}
			}

			imshow( "transformed", transformed_image );
			//imwrite("~/ejemplowrite.png",transformed_image );

			cv::Mat result;

			warpPerspective(firstImgColor, result, H,
					cv::Size(2000, this->firstImg.rows));
			cv::Mat half(result, cv::Rect(0, 0, this->secondImg.cols, this->secondImg.rows));
			secondImgColor.copyTo(half);
			result.copyTo(secondImgColor);

			imshow("Easy Merge Result", result);

			cv::waitKey(3);
			//imwrite("~/result.png",result);
			std::swap(firstImgColor, secondImgColor);
			std::swap(firstImg, secondImg);



			std::stringstream ss;
			ss << "Processed frames --> " << this->numberOfProcessedFrames << ", Keypoints -->" << this->numberKeypoints << ", Matches-->" << this->numberMatches << ", GoodMatches-->" << this->numberGoodMatches;

			std::cout << ss.str().c_str() << std::endl;
		}
	}

private:
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;

	float filter1;
	float filter2;
	DETECTOR detector;
	EXTRACTOR extractor;
	bool first;

	Mat firstImg;
	Mat firstImgColor;
	Mat secondImg;
	Mat secondImgColor;

	int numberOfProcessedFrames;
	int numberKeypoints;
	int numberMatches;
	int numberGoodMatches;

};

int main(int argc, char** argv) {
	time_t inicio, fin;
	ros::init(argc, argv, "image_converter");
	inicio = time(NULL);
	PanoramCreator ic(PanoramCreator::bag, PanoramCreator::SIFT_D, PanoramCreator::SIFT_E, 0.8, 2);
	ros::spin();
	fin = time(NULL);
	std::cout << "tiempo tardado: " << fin-inicio;
	return 0;
}
