% 读取txt文件
data1 = importdata('1030RW_AfterCal_30min_2-1.txt');
data2 = importdata('1030RW_AfterCal_30min_2-2.txt');
data3 = importdata('1030RW_AfterCal_30min_2-3.txt');
data4 = importdata('1030RW_AfterCal_30min_2-4.txt');
data5 = importdata('1030RW_AfterCal_30min_2-5.txt');
data6 = importdata('1030RW_AfterCal_30min_2-6.txt');


% 解析数据
quaternions1 = data1(:,1:4);
euler_angles1 = data1(:,5:7);
quaternions2 = data2(:,1:4);
euler_angles2 = data2(:,5:7);
quaternions3 = data3(:,1:4);
euler_angles3 = data3(:,5:7);
quaternions4 = data4(:,1:4);
euler_angles4 = data4(:,5:7);
quaternions5 = data5(:,1:4);
euler_angles5 = data5(:,5:7);
quaternions6 = data6(:,1:4);
euler_angles6 = data6(:,5:7);

% % 通过平均值得到真实值的估计
% estimated_real_quaternion = mean(quaternions);
% estimated_real_euler = mean(euler_angles);
% 
% % 计算漂移误差
% % drift_error_quaternion = quatmultiply(quaternions, quatinv(estimated_real_quaternion));
% % drift_error_euler = euler_angles - estimated_real_euler;
% 
% % 计算方差
% quat_variance = var(quaternions);
% euler_variance = var(euler_angles);

% 绘图
figure(1);
set(gcf, 'Color', [1, 1, 1],'Position', [-974 -40 500 500]);
subplot(3,2,1);
plot(quaternions1);
title('Quaternion Pose1');
subplot(3,2,2);
plot(quaternions2);
title('Quaternion Pose2');
subplot(3,2,3);
plot(quaternions3);
title('Quaternion Pose3');
subplot(3,2,4);
plot(quaternions4);
title('Quaternion Pose4');
subplot(3,2,5);
plot(quaternions5);
title('Quaternion Pose5');
subplot(3,2,6);
plot(quaternions6);
title('Quaternion Pose6');

%% imu noise estimation
% 一阶多项式拟合
% x = 1:length(quaternions); % 创建x轴数据
% p_roll = polyfit(x, euler_angles(:,1), 1); % 对欧拉角数据进行一阶多项式拟合
% p_pitch = polyfit(x, euler_angles(:,2), 1);
% p_yaw = polyfit(x, euler_angles(:,3), 1);


%AD Var
% [avar1,tau1] = allanvar(euler_angles(:,1));
% [avar2,tau2] = allanvar(euler_angles(:,2));
% [avar3,tau3] = allanvar(euler_angles(:,3));
% 
% figure(2);
% subplot(3,1,1);
% loglog(tau1,avar1)
% xlabel('\tau')
% ylabel('\sigma^2(\tau)')
% title('Allan Variance for Roll')
% grid on
% 
% subplot(3,1,2);
% loglog(tau2,avar2)
% xlabel('\tau')
% ylabel('\sigma^2(\tau)')
% title('Allan Variance for Pitch')
% grid on
% 
% subplot(3,1,3);
% loglog(tau3,avar3)
% xlabel('\tau')
% ylabel('\sigma^2(\tau)')
% title('Allan Variance for Yaw')
% grid on
% 
% 
% disp('四元数方差:');
% disp(quat_variance);
% disp('欧拉角方差:');
% disp(euler_variance);
% disp('欧拉角一阶多项式拟合系数:');
% disp(p_roll);
% disp(p_pitch);
% disp(p_yaw);