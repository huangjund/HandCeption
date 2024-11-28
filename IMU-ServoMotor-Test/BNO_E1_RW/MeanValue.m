% 读取txt文件
data = importdata('1030RW_AfterCal_30min_2-6.txt');

% 解析数据
quaternions = data(:,1:4);
euler_angles = data(:,5:7);


estimated_real_quaternion = mean(quaternions);
estimated_real_euler = mean(euler_angles);

disp('Q_mean:');
disp(estimated_real_quaternion);
disp('Euler_mean:');
disp(estimated_real_euler);