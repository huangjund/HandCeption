% 读取保存好的IMU数据文件
data = importdata('test1.txt');

% 提取加速度计、陀螺仪和磁力计数据
accData = data.data(2:end,3:5);
gyroData = data.data(2:end,6:8);
magData = data.data(2:end,9:11);
eulerangleData = data.data(2:end,12:14);

% 从数据中提取时间间隔
N = size(data.data,1) -1;
time = (0:(N-1))'/100; % 100Hz

% 初始化AHRS滤波器
filter1 = ahrsfilter('SampleRate', 100, 'GyroscopeNoise', 0.1, 'AccelerometerNoise', 0.1, 'MagnetometerNoise', 0.1);

% 初始化insEKF
motionModel = insMotionPose;
measureNoise = struct("AccelerometerNoise", 0.1, ...
    "GyroscopeNoise", 1, ...
    "MagnetometerNoise", 0.1);
timeStamp = seconds(0:size(accData)-1)/100; % 创建时间戳
sensorData = timetable(timeStamp',  accData, gyroData, magData, 'VariableNames', {'Accelerometer', 'Gyroscope', 'Magnetometer'});
accel = insAccelerometer;
gyro = insGyroscope;
mag = insMagnetometer;
filter2 = insEKF(accel, gyro, mag, motionModel);

stateparts(filt,"Orientation",compact(initOrient));
statecovparts(filt,"Orientation",1e-3);
% estOrientation = quaternion.zeros(N,1);
estPosition = zeros(N,3);

for i=1:N
    predict(filter2, 0.01); % Ts sample time
    fuse(filter2, accel, accData(i, :), 0.1);
    fuse(filter2, gyro, gyroData(i, :), 1);
    fuse(filter2, mag, magData(i, :), 0.1);
    estPosition(i, :) = stateparts(filter2,"Position");
    % estOrientation(i) = quaternion(stateparts(filter2,"Orientation"));
end

[estimates,smoothEstimates] = estimateStates(filter2,sensorData,measureNoise);

% Quat to Euler
EulerAngles_EKF = eulerd(smoothEstimates.Orientation,"ZYX","frame");

% 运行AHRS滤波器
[orientation, angularVelocity] = filter1(accData, gyroData, magData);

% Euler Angle from MTi
EulerAngles_GT = eulerangleData;
% Euler Angle Estimated by AHRSfilt
EulerAngles_Est = eulerd(orientation,'ZYX','frame');
% Yaw oringin
EulerAngles_Est(:,1) = EulerAngles_Est(:,1)-EulerAngles_Est(1,1)+EulerAngles_GT(1,3);
EulerAngles_EKF(:,1) = EulerAngles_EKF(:,1)-EulerAngles_EKF(1,1)+EulerAngles_GT(1,3);

% 可视化
figure(1);
subplot(3,1,1)
plot(time,accData)
title('Accelerometer Reading')
ylabel('Acceleration (m/s^2)')

subplot(3,1,2)
plot(time,gyroData)
title('Magnetometer Reading')
ylabel('Magnetic Field (\muT)')

subplot(3,1,3)
plot(time,magData)
title('Gyroscope Reading')
ylabel('Angular Velocity (rad/s)')
xlabel('Time (s)')

figure(2);
subplot(3,1,1)
plot(time, EulerAngles_Est(:,1), time,EulerAngles_GT(:,3), time, EulerAngles_EKF(:,1))
title('Rotation around z-axis/ Yaw')
ylabel('Rotation (degrees)')
xlabel('Time (s)')
legend('AHRS','MTi-algo','EKF')

subplot(3,1,2)
plot(time, EulerAngles_Est(:,2), time,EulerAngles_GT(:,2), time, EulerAngles_EKF(:,2))
title('Rotation around y-axis/ Pitch')
ylabel('Rotation (degrees)')
xlabel('Time (s)')
legend('AHRS','MTi-algo','EKF')

subplot(3,1,3)
plot(time, EulerAngles_Est(:,3), time,EulerAngles_GT(:,1), time, EulerAngles_EKF(:,3))
title('Rotation around x-axis/ Roll')
ylabel('Rotation (degrees)')
xlabel('Time (s)')
legend('AHRS','MTi-algo','EKF')

figure;
h = plot3(estPosition(1, 1), estPosition(1, 2), estPosition(1, 3), 'LineWidth', 1.5);
grid on;
for i = 2:N
    % 更新轨迹位置
    set(h, 'XData', estPosition(1:i, 1), 'YData', estPosition(1:i, 2), 'ZData', estPosition(1:i, 3));
    drawnow; % 实时绘制
    pause(0.1); % 可以根据需要调整更新间隔
end

% 保存位置数据
% output_position_file = 'output_position.txt';
% dlmwrite(output_position_file, pose, 'delimiter', '\t');