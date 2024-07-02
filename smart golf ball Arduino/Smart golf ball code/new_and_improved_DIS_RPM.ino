
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
int max_decel_x = 0;
int max_decel_y = 0;
int max_decel_z = 0;
int prev_accel_x = 0;
int prev_accel_y = 0;
int prev_accel_z = 0;
float current_accel_x;
float current_accel_y;
float current_accel_z;
float delta_accel_x;
float delta_accel_y;
float delta_accel_z;

void gyro_signals(void) {
  Wire.beginTransmission(0x68);
  Wire.write(0x1A);
  Wire.write(0x05);
  Wire.endTransmission();
 // Wire.beginTransmission(0x68);
 // Wire.write(0x1C);
 // Wire.write(0x10);
 // Wire.endTransmission();
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

  AccX=(float)AccXLSB/16384;
  AccY=(float)AccYLSB/16384;
  AccZ=(float)AccZLSB/16384;
 // AccX=(float)AccXLSB/4096;
  //AccY=(float)AccYLSB/4096;
  //AccZ=(float)AccZLSB/4096;
  rawroll = atan2(AccY, AccZ) * 180 / PI;
  // Calculate pitch angle
  rawpitch = atan2(AccX, AccZ) * 180 / PI;
  rawyaw = atan2(AccX, AccY) * 180 / PI;
}
void setup() {

  Serial.begin(115200);
  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH);
  Wire.setClock(400000);
  Wire.begin();
  delay(250);
  Wire.beginTransmission(0x68); 
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission();
}
void loop() {
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
    //clegg impact tester
    /////////////////////////////////////////////////////////////////////////////
    
        delta_accel_x = (AccX*9.81) - prev_accel_x;
        delta_accel_y = (AccY*9.81) - prev_accel_y;
        delta_accel_z = (AccZ*9.81) - prev_accel_z;
        if (delta_accel_x < 0) {
          if (delta_accel_x < max_decel_x) {
              max_decel_x = delta_accel_x;
          }
        }
        if (delta_accel_y < 0) {
          if (delta_accel_y < max_decel_y){
              max_decel_y = delta_accel_y;
          }
        }
        if (delta_accel_z < 0) {
          if (delta_accel_z < max_decel_z) {
              max_decel_z = delta_accel_z;
          }
        }
        prev_accel_x = (AccX*9.81);
        prev_accel_y = (AccY*9.81);
        prev_accel_z = (AccY*9.81); 

/////////////////////////////////////////////////////////////////////////////////////
 //angle_difference_Z = Current_angle_Z-previous_angle_Z;
     Serial.print("Acceleration X [g]= ");
     Serial.print(AccX);
     Serial.print(" Acceleration Y [g]= ");
     Serial.print(AccY);
     Serial.print(" Acceleration Z [g]= ");
     Serial.print(AccZ);
     //Serial.print("Delta_roll : ");
	  // Serial.print(angle_difference_roll);
	   //Serial.print("Delta_pitch : ");
	   //Serial.print(angle_difference_pitch);
        Serial.print("\tmax_decel_x:");
        Serial.print(max_decel_x);
        Serial.print("\tmax_decel_y:");
        Serial.print(max_decel_y);
        Serial.print("\tmax_decel_z:");
        Serial.println(max_decel_z);
      
  /*Serial.print("angle roll= ");
  Serial.print(AngleRoll);
  Serial.print("\tangle pitch= ");
  Serial.print(AnglePitch);
   Serial.print("\tangle yaw= ");
  Serial.print(AngleYaw);
  //Serial.print("AVR : ");
	//Serial.print(angular_speed_roll);
	//Serial.print("AVP : ");
	//Serial.print(angular_speed_pitch);
  //Serial.print("\tRPSX : ");
	//Serial.print(RPS_roll);
	//Serial.print("\tRPSY : ");
	//Serial.print(RPS_pitch);
  Serial.print("\tdistance_roll : ");
	Serial.print(distance_roll);
	Serial.print("\tdistance_pitch : ");
	Serial.print(distance_pitch);
  Serial.print("\tdistance_yaw : ");
	Serial.print(distance_yaw);*/
   
  sample_rate = 1000/(delta_time);
     

  Serial.print("\telapsed time: " + String(deltaTimeSeconds, 4) + " s");
  Serial.println("\tSample Rate: " + String(sample_rate) + " Hz");
  previous_time = current_time;
  previous_angle_roll= AngleRoll;
  previous_angle_pitch=AnglePitch;
  previous_angle_yaw=AngleYaw;
}