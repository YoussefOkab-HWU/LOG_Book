#include <WiFi.h>

#include <WebSocketClient.h>

const char* ssid     = "iPhone";
const char* password = "Yo112453";
char path[] = "/";
char host[] = "172.20.10.8:5353";
  
WebSocketClient webSocketClient;
//unsigned long current_time;
// Use WiFiClient class to create TCP connections
WiFiClient client;
#include <cmath>
#include <Wire.h>
float RateRoll, RatePitch, RateYaw;
float AccX, AccY, AccZ;
float AngleRoll, AnglePitch,AngleYaw, rawroll, rawpitch, rawyaw;
float LoopTimer;
float angle_difference_roll, angle_difference_pitch, angle_difference_yaw;
float angular_speed_roll, angular_speed_pitch, angular_speed_yaw;
float RPS_roll, RPS_pitch, RPS_yaw;
float distance_roll, distance_pitch, distance_yaw;
float previous_angle_roll, previous_angle_pitch, previous_angle_yaw;
unsigned long previous_time;
unsigned long current_time;
unsigned long delta_time;
unsigned long sample_rate;
/*#include <i2cdetect.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include <MPU6050_light.h>
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif
MPU6050_6Axis_MotionApps20 mpu;
MPU6050 mpu1(Wire);
unsigned long previous_time;

unsigned long delta_time;
unsigned long sample_rate;
//#define OUTPUT_READABLE_WORLDACCEL

//#define OUTPUT_READABLE_angular_speed
#define INTERRUPT_PIN 2  // use pin 2 on Arduino Uno & most boards
#define LED_PIN 13 // (Arduino is 13, Teensy is 11, Teensy++ is 6)
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer
uint32_t LoopTimer;
uint32_t TotalTimer;
bool rolling = false; // state if the ball is rolling
bool started = false; // state if the ball started rolling
bool first_time_flag = true;
VectorInt16 gyro;

float aaWorldms2[3];    // [x, y, z]            world-frame accel sensor measurements in m.s²
#include <cmath>

#define PI 3.1415926535897932384626433832795

VectorFloat gravity;    // [x, y, z]            gravity vector
float aaWorldNorm;      //                      norm of the world acceleration
float euler[3];         // [psi, theta, phi]    Euler angle container
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
float AccelerationX;
float AccelerationY;
float AccelerationZ;
float speedX; // We use separated floats rather than a table because we didn't manage to send the values of the table by BLE (seems to send the adresses of the values)
float speedY;
float speedZ;
float displacementX; // We use separated floats rather than a table because we didn't manage to send the values of the table by BLE (seems to send the adresses of the values)
float displacementY;
float displacementZ;
float angle_difference_X;
float angle_difference_Y;
float angle_difference_Z;
float Current_angle_X ;
float Current_angle_Y ;
float Current_angle_Z ;
double angular_speedX;
double angular_speedY;
double angular_speedZ;
float RPMX;
float RPMY;
float RPMZ;
double RPSX;
double RPSY;
double RPSZ;
double distanceX;
double distanceY;
double distanceZ;
double  abs_angle_difference_X;
double  abs_angle_difference_Y;
double  abs_angle_difference_Z;

//  ||BLUETOOTH||
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLEAdvertising.h>
#include <BLE2902.h>

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
bool bluetoothWaitingMessagePrinted = false;
bool bluetoothConnectedPrinted = false;
bool ballisstillprinted = false;
bool Rollingprinted = false;
bool senddata = false;
bool sky = true;
uint32_t value = 0;
float previous_angle_X;
float previous_angle_Y;
float previous_angle_Z;
//uint32_t Current_angle_X = 0;
//uint32_t Current_angle_Y = 0;
//uint32_t Current_angle_Z = 0;
double accelerationThreshold = 0.4; // Adjust based on your specific application and sensor characteristics
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

volatile bool mpuInterrupt = false;     // indicates whether MPU interrupt pin has gone high
void dmpDataReady() {
    mpuInterrupt = true;
}
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
       deviceConnected = true;
      // bluetoothConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
       deviceConnected = false;
      // bluetoothConnected = false;
    }
};*/
void gyro_signals(void) {
  Wire.beginTransmission(0x68);
  Wire.write(0x1A);
  Wire.write(0x05);
  Wire.endTransmission();
  Wire.beginTransmission(0x68);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission();
  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  Wire.endTransmission(); 
  Wire.requestFrom(0x68,6);
  int16_t AccXLSB = Wire.read() << 8 | Wire.read();
  int16_t AccYLSB = Wire.read() << 8 | Wire.read();
  int16_t AccZLSB = Wire.read() << 8 | Wire.read();
  Wire.beginTransmission(0x68);
  Wire.write(0x1B); 
  Wire.write(0x8);
  Wire.endTransmission();                                                   
  Wire.beginTransmission(0x68);
  Wire.write(0x43);
  Wire.endTransmission();
  Wire.requestFrom(0x68,6);
  /*int16_t GyroX=Wire.read()<<8 | Wire.read();
  int16_t GyroY=Wire.read()<<8 | Wire.read();
  int16_t GyroZ=Wire.read()<<8 | Wire.read();
  RateRoll=(float)GyroX/65.5;
  RatePitch=(float)GyroY/65.5;
  RateYaw=(float)GyroZ/65.5;*/
  AccX=(float)AccXLSB/16384;
  AccY=(float)AccYLSB/16384;
  AccZ=(float)AccZLSB/16384;
  //AngleRoll=atan(AccY/sqrt(AccX*AccX+AccZ*AccZ))*1/(3.142/180);
  //AnglePitch=-atan(AccX/sqrt(AccY*AccY+AccZ*AccZ))*1/(3.142/180);
  rawroll = atan2(AccY, AccZ) * 180 / PI;

  // Calculate pitch angle
  rawpitch = atan2(AccX, AccZ) * 180 / PI;
  rawyaw = atan2(AccX, AccY) * 180 / PI;
}
void setup() {

  Serial.begin(115200);
  delay(10);

  // We start by connecting to a WiFi network

  Serial.println();
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");  
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  delay(5000);
  

  // Connect to the websocket server
  if (client.connect("172.20.10.8", 5353)) {
    Serial.println("Connected");
  } else {
    Serial.println("Connection failed.");
    Serial.println(WiFi.status());
    while(1) {
      // Hang on failure
    }
  }

  // Handshake with the server
  webSocketClient.path = path;
  webSocketClient.host = host;
  if (webSocketClient.handshake(client)) {
    Serial.println("Handshake successful");
  } else {
    Serial.println("Handshake failed.");
    while(1) {
      // Hang on failure
    }  
  }
// put your setup code here, to run once:
   // join I2C bus (I2Cdev library doesn't do this automatically)
    //#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        //Wire.begin();
      //  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
    //#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
       // Fastwire::setup(400, true);
   // #endif

    // initialize serial communication
    // (115200 chosen because it is required for Teapot Demo output, but it's
    // really up to you depending on your project)
    Serial.begin(115200);
    
    //BLEDevice::init("Smart Golf Ball");
   //BLEDevice::init("ESP32");
    //BLEDevice::setPower(ESP_PWR_LVL_P9);

      pinMode(13, OUTPUT);
      digitalWrite(13, HIGH);
      Wire.setClock(400000);
      Wire.begin();
      delay(250);
      Wire.beginTransmission(0x68); 
      Wire.write(0x6B);
      Wire.write(0x00);
      Wire.endTransmission();
    // Create the BLE Server
   // pServer = BLEDevice::createServer();
    //pServer->setCallbacks(new MyServerCallbacks());

    // Create the BLE Service
    /*BLEService *pService = pServer->createService(SERVICE_UUID);
    
    // Create a BLE Characteristic
    pCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID,
                        BLECharacteristic::PROPERTY_READ   |
                        BLECharacteristic::PROPERTY_WRITE  |
                        BLECharacteristic::PROPERTY_NOTIFY |
                        BLECharacteristic::PROPERTY_INDICATE
                      );

    // https://www.bluetooth.com/specifications/gatt/viewer?attributeXmlFile=org.bluetooth.descriptor.gatt.client_characteristic_configuration.xml
    // Create a BLE Descriptor
    pCharacteristic->addDescriptor(new BLE2902());

    // Start the service
    pService->start();

    // Start advertising
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    //pAdvertising->setMinInterval(0x0B20);
    //pAdvertising->setMaxInterval(0x0C80);
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0);  // set value to 0x00 to not advertise this parameter
    BLEDevice::startAdvertising();
    Serial.println("Waiting a client connection to notify...");
 
    while (!Serial); // wait for Leonardo enumeration, others continue immediately

    // NOTE: 8MHz or slower host processors, like the Teensy @ 3.3V or Arduino
    // Pro Mini running at 3.3V, cannot handle this baud rate reliably due to
    // the baud timing being too misaligned with processor ticks. You must use
    // 38400 or slower in these cases, or use some kind of external separate
    // crystal solution for the UART timer.

    // initialize device
    Serial.println(F("Initializing I2C devices..."));
    mpu.initialize();
    pinMode(INTERRUPT_PIN, INPUT);

    // verify connection
    Serial.println(F("Testing device connections..."));
    Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));

    
    // load and configure the DMP
    Serial.println(F("Initializing DMP..."));
    devStatus = mpu.dmpInitialize();

    Serial.println(mpu.getFullScaleAccelRange());

    // supply your own gyro offsets here, scaled for min sensitivity
    mpu.setXGyroOffset(220);
    mpu.setYGyroOffset(76);
    mpu.setZGyroOffset(-85);
    mpu.setZAccelOffset(1788); // 1688 factory default for my test chip

    // Some time to place the ball in the original position (Gravity along z)


    Serial.println("Waiting for the set up (5s)");
    digitalWrite(2, HIGH);   // Turn the RGB LED white
    delay(5000);
    digitalWrite(2, LOW);    // Turn the RGB LED off
    Serial.println("GO");
    
    // make sure it worked (returns 0 if so)
    if (devStatus == 0) {
        // Calibration Time: generate offsets and calibrate our MPU6050
        mpu.CalibrateAccel(6);
        mpu.CalibrateGyro(6);
        mpu.PrintActiveOffsets();
        // turn on the DMP, now that it's ready
        Serial.println(F("Enabling DMP..."));
        mpu.setDMPEnabled(true);

        // enable Arduino interrupt detection
        Serial.print(F("Enabling interrupt detection (Arduino external interrupt "));
        Serial.print(digitalPinToInterrupt(INTERRUPT_PIN));
        Serial.println(F(")..."));
        attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
        mpuIntStatus = mpu.getIntStatus();

        // set our DMP Ready flag so the main loop() function knows it's okay to use it
        Serial.println(F("DMP ready! Waiting for first interrupt..."));
        dmpReady = true;

        // get expected DMP packet size for later comparison
        packetSize = mpu.dmpGetFIFOPacketSize();
    } else {
        // ERROR!
        // 1 = initial memory load failed
        // 2 = DMP configuration updates failed
        // (if it's going to break, usually the code will be 1)
        Serial.print(F("DMP Initialization failed (code "));
        Serial.print(devStatus);
        Serial.println(F(")"));
    }

    // Initialisation of the variables
   // for (int i = 0; i < 3; i++) {
     // speed[i] = 0.000;
    //  displacement[i] = 0.000;
    //}
    speedX = 0.00;
    speedY = 0.00;
    speedZ = 0.00;

    displacementX = 0.00;
    displacementY = 0.00;
    displacementZ = 0.00;
    Serial.begin(115200);
  Wire.begin();
  
  byte status = mpu1.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status!=0){ } // stop everything if could not connect to MPU6050
  
  Serial.println(F("Calculating offsets, do not move MPU6050"));
  delay(1000);
  // mpu1.upsideDownMounting = true; // uncomment this line if the MPU6050 is mounted upside-down
  mpu1.calcOffsets(); // gyro and accelero
  Serial.println("Done!\n");

    // || INITALISATION OF THE BLUETOOTH COMMUNICATION ||
    // Create the BLE Device
    // previous_angle_X = 0;
    // previous_angle_Y = 0;
    // previous_angle_Z = 0;
    // previous_time = millis();
*/

}


void loop() {
  String data;

  if (client.connected()) {
     current_time = millis(); 
  /*mpu1.update();
  delta_time = current_time - previous_time; 
  Current_angle_X = mpu1.getAngleX();
  Current_angle_Y = mpu1.getAngleY();
  Current_angle_Z = mpu1.getAngleZ();
  angle_difference_X = Current_angle_X-previous_angle_X;
  angle_difference_Y = Current_angle_Y-previous_angle_Y;
  angle_difference_Z = Current_angle_Z-previous_angle_Z;
  abs_angle_difference_X = abs(Current_angle_X-previous_angle_X);
  abs_angle_difference_Y = abs(Current_angle_Y-previous_angle_Y);
  abs_angle_difference_Z = abs(Current_angle_Z-previous_angle_Z);
  previous_angle_X = Current_angle_X;
  previous_angle_Y =  Current_angle_Y;
  previous_angle_Z = Current_angle_Z;
          
  //if((millis()-timer)>10){ // print data every 10ms
  angular_speedX = (angle_difference_X/delta_time);
  angular_speedY = (angle_difference_Y/delta_time);
  angular_speedZ = (angle_difference_Z/delta_time);
  //RPMX = (1/6)*angular_speedX;
  //RPMY = (1/6)*angular_speedY;
  //RPMZ = (1/6)*angular_speedZ;
  RPSX = (angular_speedX/360)*1000;
  RPSY = (angular_speedY/360)*1000;
  RPSZ = (angular_speedZ/360)*1000;
  float deltaTimeSeconds = delta_time / 1000.0;
  
 // distanceX = (RPMX * 2 * PI * 0.028) * (delta_time / 60000);
  //distanceY = (RPMY * 2 * PI * 0.028) * (delta_time / 60000);
  //distanceZ = (RPMZ * 2 * PI * 0.028) * (delta_time / 60000);
  //distanceX += (RPSX * 2 * PI * 0.0335); 
  //distanceY += (RPSY * 2 * PI * 0.0335);
  //distanceZ += (RPSZ * 2 * PI * 0.0335); 
  if (abs_angle_difference_X > 0.05){
  distanceX += (RPSX * 2 * PI * 0.0335)*deltaTimeSeconds; 
  }else{
  distanceX = distanceX;
  }
   if (abs_angle_difference_Y > 0.05){
  distanceY += (RPSY * 2 * PI * 0.0335)*deltaTimeSeconds; 
  }else{
  distanceY = distanceY;
  }
  if (abs_angle_difference_Z > 0.05){
  distanceZ += (RPSZ * 2 * PI * 0.0335)*deltaTimeSeconds; 
  }else{
    distanceZ = distanceZ;
  }*/
     current_time = millis(); 
 // mpu1.update();
  delta_time = current_time - previous_time; 
  float deltaTimeSeconds = delta_time / 1000.0;  
  gyro_signals();
  if (rawpitch >= 0){
    AnglePitch = rawpitch;
  }
  else{
    AnglePitch = 360 + rawpitch;
  }
  if (rawroll >= 0){
    AngleRoll = rawroll;
  }else{
    AngleRoll = 360 + rawroll;
  }
   if (rawyaw >= 0){
    AngleYaw = rawyaw;
  }else{
    AngleYaw = 360 + rawyaw;
  }
  angle_difference_roll = AngleRoll-previous_angle_roll;
  angle_difference_pitch = AnglePitch-previous_angle_pitch;
  angle_difference_yaw = AngleYaw-previous_angle_yaw;
  if (angle_difference_roll > 180.0) {
        angle_difference_roll -= 360.0;
    } else if (angle_difference_roll < -180.0) {
        angle_difference_roll += 360.0;
    }
     if (angle_difference_pitch > 180.0) {
        angle_difference_pitch -= 360.0;
    } else if (angle_difference_pitch< -180.0) {
        angle_difference_pitch += 360.0;
    }
     if (angle_difference_yaw > 180.0) {
        angle_difference_yaw -= 360.0;
    } else if (angle_difference_yaw< -180.0) {
        angle_difference_yaw += 360.0;
    }
    angular_speed_roll = (angle_difference_roll/deltaTimeSeconds);
    angular_speed_pitch = (angle_difference_pitch/deltaTimeSeconds);
    angular_speed_yaw = (angle_difference_yaw/deltaTimeSeconds);
    RPS_roll = (angular_speed_roll/360);
    RPS_pitch = (angular_speed_pitch/360);
    RPS_yaw = (angular_speed_yaw/360);
    distance_roll += (RPS_roll * PI * 0.04268)*deltaTimeSeconds; 
    distance_pitch += (RPS_pitch * PI * 0.04268)*deltaTimeSeconds; 
    distance_yaw += (RPS_yaw * PI * 0.04268)*deltaTimeSeconds; 
 // angle_difference_Z = Current_angle_Z-previous_angle_Z;
   //Serial.print("Acceleration X [g]= ");
   //Serial.print(AccX);
   //Serial.print(" Acceleration Y [g]= ");
   //Serial.print(AccY);
   //Serial.print(" Acceleration Z [g]= ");
   //Serial.print(AccZ);
  //Serial.print("Delta_roll : ");
	//Serial.print(angle_difference_roll);
	//Serial.print("Delta_pitch : ");
	//Serial.print(angle_difference_pitch);
 /* Serial.print("roll angle= ");
  Serial.print(AngleRoll);
  Serial.print(" angle pitch= ");
  Serial.print(AnglePitch);
   Serial.print(" angle yaw= ");
  Serial.print(AngleYaw);
  //Serial.print("AVR : ");
	//Serial.print(angular_speed_roll);
	//Serial.print("AVP : ");
	//Serial.print(angular_speed_pitch);
  //Serial.print("\tRPSX : ");
	//Serial.print(RPS_roll);
	//Serial.print("\tRPSY : ");
	//Serial.print(RPS_pitch);
  Serial.print("distance_roll : ");
	Serial.print(distance_roll);
	Serial.print("distance_pitch : ");
	Serial.print(distance_pitch);
  Serial.print("distance_yaw : ");
	Serial.print(distance_yaw);
   
  sample_rate = 1000/(delta_time);
     

  Serial.print("elapsed time: " + String(deltaTimeSeconds, 4) + " s");
  Serial.println("Sample Rate: " + String(sample_rate) + " Hz");*/
  previous_time = current_time;
  previous_angle_roll= AngleRoll;
  previous_angle_pitch=AnglePitch;
  previous_angle_yaw=AngleYaw;
    webSocketClient.getData(data);
    if (data.length() > 0) {
      //Serial.print("Received data: ");
      Serial.println(data);
    }
    //data = "Hey Val!!";
    data = "angle roll= " + String(AngleRoll) + "angle pitch= "+ String(AnglePitch) + "angle yaw=  "+ String(AngleYaw) + "distance_roll : " + String(distance_roll) + "distance_pitch :  " + String(distance_pitch)+ "distance_yaw :" + String(distance_yaw);
    
    webSocketClient.sendData(data);
    
  } else {
    Serial.println("Client disconnected.");
    while (1) {
      // Hang on disconnect.
    }
  }
  
  // wait to fully let the client disconnect
  delay(30);
  
}