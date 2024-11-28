%% (Main) Configure Servos
clear
close all
clc
% Set COM port
com_port = 'COM5';
% Set baudrate
baud_rate = 57600;
% Set protocol Version
protocol_version = 2;
% Initialize library and port number
[lib_name, port_num] = initDxl(com_port);
% Open port
openPortDxl(lib_name, port_num);
% Set Baudrate
setBaudDxl(lib_name, port_num, baud_rate);
% Find dynamixels
[motor_IDs, motor_models] = findDxl(lib_name, port_num, protocol_version);
% Operating modes
operatingModeDxl(lib_name, port_num, protocol_version, motor_IDs, ...
    motor_models, 'Extended Position Control');
% Drive modes
driveModeDxl(lib_name, port_num, protocol_version, motor_IDs, ...
    motor_models, 'Velocity-based Profile');
% % Enable torque
% torqueDxl(lib_name, port_num, protocol_version, motor_IDs, ...
%     motor_models, 'Enable');
% Check error
checkErrorDxl(lib_name, port_num, protocol_version)

%% (Main) Configure IMU
% Launch activex server
% define and initialize global variable
global h IMUData;
IMUData={0,0,[1,0,0,0]};
try
    switch computer
        case 'PCWIN'
            h = actxserver('xsensdeviceapi_com32.IXsensDeviceApi');
        case 'PCWIN64'
            h = actxserver('xsensdeviceapi_com64.IXsensDeviceApi');
        otherwise
            error('XDA:os','Unsupported OS');
    end
catch e
    fprintf('\nPlease reinstall MT SDK or check manual,\n Xsens Device Api is not found.')
    rethrow(e);
end
fprintf('\nActiveXsens server - activated \n');

version = h.xdaVersion;
fprintf('Using XDA version: %.0f.%.0f.%.0f\n',version{1:3})
if length(version)>3
    fprintf('XDA build: %.0f%s\n',version{4:5})
end

% Scan for devices
fprintf('Scanning for devices... \n')
ports = h.XsScanner_scanPorts(0,100,true,true);

% Find an MTi device
numPorts = size(ports,1);
for port = 1:numPorts
    if (h.XsDeviceId_isMti(ports{port,1}) || h.XsDeviceId_isMtig(ports{port,1}))
        mtPort = ports(port,:);
        break
    end
end

if isempty(mtPort)
    fprintf('No MTi device found. Aborting. \n');
    return
end

deviceId = mtPort{1};
portName = mtPort{3};
baudrate = mtPort{4};

fprintf('Found a device with: \n');
fprintf(' Device ID: %s \n', h.XsDeviceId_toDeviceTypeString(deviceId, false));
fprintf(' Baudrate: %d \n', baudrate);
fprintf(' Port name: %s \n', portName);


% Open port
fprintf('Opening port... \n')
if ~h.XsControl_openPort(portName, baudrate, 0, true)
    fprintf('Could not open port. Aborting. \n');
    return
end

% Get the device object
device = h.XsControl_device(deviceId);
fprintf('Device: %s, with ID: %s opened. \n', h.XsDevice_productCode(device), h.XsDeviceId_toString(h.XsDevice_deviceId(device)));


% Register eventhandler
h.registerevent({'onLiveDataAvailable',@eventhandlerXsens});
h.setCallbackOption(h.XsComCallbackOptions_XSC_LivePacket, h.XsComCallbackOptions_XSC_None);
% show events using h.events and h.eventlisteners too see which are registerd;

% Put device into configuration mode
fprintf('Putting device into configuration mode... \n')
if ~h.XsDevice_gotoConfig(device)
    fprintf('Could not put device into configuration mode. Aborting. \n');
    return
end

% Configure the device
fprintf('Configuring the device... \n')
if (h.XsDeviceId_isImu(deviceId))
    outputConfig = {h.XsDataIdentifier_XDI_PacketCounter,0;
        h.XsDataIdentifier_XDI_SampleTimeFine,0;
        h.XsDataIdentifier_XDI_Acceleration,100;
        h.XsDataIdentifier_XDI_RateOfTurn,100;
        h.XsDataIdentifier_XDI_MagneticField,100};
elseif (h.XsDeviceId_isVru(deviceId) || h.XsDeviceId_isAhrs(deviceId))
    outputConfig = {h.XsDataIdentifier_XDI_PacketCounter,0;
        h.XsDataIdentifier_XDI_SampleTimeFine,0;
        h.XsDataIdentifier_XDI_Quaternion,100};
elseif (h.XsDeviceId_isGnss(deviceId))
    outputConfig = {h.XsDataIdentifier_XDI_PacketCounter,0;
        h.XsDataIdentifier_XDI_SampleTimeFine,0;
        h.XsDataIdentifier_XDI_Quaternion,100;
        h.XsDataIdentifier_XDI_LatLon,100;
        h.XsDataIdentifier_XDI_AltitudeEllipsoid,100;
        h.XsDataIdentifier_XDI_VelocityXYZ,100};
else
    fprintf('Unknown device while configuring. Aborting. \n');
    return
end

if ~h.XsDevice_setOutputConfiguration(device,outputConfig)
    fprintf('Could not configure the device. Aborting. \n');
    return
end

% Put device into measurement mode
% start event loop
fprintf('Putting device into measurement mode... \n')
if ~h.XsDevice_gotoMeasurement(device)
    fprintf('Could not put device into measurement mode. Aborting. \n');
    return
end

%% visualization and validation
visualize_3d_frames(lib_name, port_num, protocol_version, motor_IDs, motor_models)

%% release servo and IMU
% Close port and XsControl object
fprintf('Closing port and XsControl object... \n');
h.XsControl_closePort(portName);
h.XsControl_close();

% Release COM-object
fprintf('Releasing COM-object... \n');
delete(h);
clear h;
fprintf('Successful exit. \n');

% end session
clc 
% Disable torque
torqueDxl(lib_name, port_num, protocol_version, motor_IDs, motor_models, ...
    'Disable');
% Check error
checkErrorDxl(lib_name, port_num, protocol_version)
% Close port 
closePortDxl(lib_name,port_num);

%% visualization functions
function visualize_3d_frames(lib_name, port_num, protocol_version, motor_IDs, motor_models)
    global IMUData;
    rotx = @(t) [1 0 0; 0 cos(t) -sin(t) ; 0 sin(t) cos(t)] ;
    rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;
    roty = @(t) [cos(t) 0 sin(t) ; 0 1 0 ; -sin(t) 0  cos(t)] ;
    % Set up the figure for real-time plotting
    fig = figure;
    axis([-25 25 -25 25 0 85]);  % Define axis limits
    grid on;
    hold on;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(3);  % 3D view
    
    % Create two sets of frames (for two different coordinate systems)
    frame1 = createFrame();
    frame2 = createFrame();
    frameIMU = createFrame();
    estframe1 = createEstFrame();
    estframe2 = createEstFrame();
    estframeIMU = createEstFrame();
    
   % Set up the quit flag and key press function
    quitFlag = false;
    set(fig, 'KeyPressFcn', @(src, event) setQuitFlag(event));

    Rxy = rotx(pi/2)*rotz(-pi/2);
    % Main loop for updating the frames based on real-time sensor data
    while ~quitFlag
        % Get Servo Input
        q = 0.088*modPosNeg(readDxl(lib_name, port_num, protocol_version, motor_IDs, motor_models, ...
                            'Present Position'));
        T12 = [rotz(deg2rad(q(1))) [0;0;28];0 0 0 1];
        T23 = [rotx(-pi/2)*rotz(deg2rad(q(2))) [0;0;20];0 0 0 1];
        T3I = [Rxy [0;-28.5;0];0 0 0 1];
        T13 = T12*T23;
        T1I = T13*T3I;
        % estimation
        try
            R1I_hat = quat2rotm(IMUData{3}');
            T1I_hat = [R1I_hat T1I(1:3,4);0 0 0 1];
            R1I_tilde = R1I_hat*rotz(pi/2)*rotx(-pi/2);
            q1_hat = atan(-R1I_tilde(1,3)/R1I_tilde(2,3))+pi;
            q2_hat = atan(R1I_tilde(3,1)/R1I_tilde(3,2));
            T12_hat = [rotz(q1_hat) [0;0;28];0 0 0 1];
            T23_hat = [rotx(-pi/2)*rotz(q2_hat) [0;0;20];0 0 0 1];
            T13_hat = T12_hat*T23_hat;


            % Update frames
            updateFrame(frame1, T12);
            updateFrame(frame2, T13);
            updateFrame(frameIMU, T1I);
            updateFrame(estframe1, T12_hat);
            updateFrame(estframe2, T13_hat);
            updateFrame(estframeIMU, T1I_hat);

            drawnow;  % Update the plot in real-time
        catch e
            fprint("something wrong with quat2rotm");
        end
        pause(0.01);  % Add a slight delay for smooth visualization
    end

    disp('Exiting loop...');
    
    % Nested function to set the quit flag
    function setQuitFlag(event)
        if strcmp(event.Key, 'q')
            quitFlag = true;
        end
    end
end

function frame = createFrame()
    % Create lines representing the axes of a coordinate frame
    frame.X = plot3([0 5], [0 0], [0 0], 'r', 'LineWidth', 2);  % X-axis (red)
    frame.Y = plot3([0 0], [0 5], [0 0], 'g', 'LineWidth', 2);  % Y-axis (green)
    frame.Z = plot3([0 0], [0 0], [0 5], 'b', 'LineWidth', 2);  % Z-axis (blue)
end

function frame = createEstFrame()
    % Create lines representing the axes of a coordinate frame
    frame.X = plot3([0 5], [0 0], [0 0], 'k', 'LineWidth', 2);  % X-axis (red)
    frame.Y = plot3([0 0], [0 5], [0 0], 'k', 'LineWidth', 2);  % Y-axis (green)
    frame.Z = plot3([0 0], [0 0], [0 5], 'k', 'LineWidth', 2);  % Z-axis (blue)
end

function updateFrame(frame, T)
    % Calculate rotation matrix from roll, pitch, yaw
    R = T(1:3,1:3);
    
    % Define the axes' endpoints in the local frame (unit vectors)
    origin = [T(1,4), T(2,4), T(3,4)];
    x_axis = R * [5; 0; 0];
    y_axis = R * [0; 5; 0];
    z_axis = R * [0; 0; 5];
    
    % Update the frame axes with the new transformed positions
    set(frame.X, 'XData', [origin(1) origin(1)+x_axis(1)], 'YData', [origin(2) origin(2)+x_axis(2)], 'ZData', [origin(3) origin(3)+x_axis(3)]);
    set(frame.Y, 'XData', [origin(1) origin(1)+y_axis(1)], 'YData', [origin(2) origin(2)+y_axis(2)], 'ZData', [origin(3) origin(3)+y_axis(3)]);
    set(frame.Z, 'XData', [origin(1) origin(1)+z_axis(1)], 'YData', [origin(2) origin(2)+z_axis(2)], 'ZData', [origin(3) origin(3)+z_axis(3)]);
end

function pose = getRealTimePoseFromSensor1()
    % Replace this with code to read real-time data from sensor 1
    % Simulate a moving pose
    t = toc;  % Use time to simulate changing poses
    pose = [5*sin(t), 5*cos(t), 5, 0.1*sin(t), 0.2*cos(t), 0.3*sin(t)];  % Example pose
end

function pose = getRealTimePoseFromSensor2()
    % Replace this with code to read real-time data from sensor 2
    % Simulate a second moving pose
    t = toc;  % Use time to simulate changing poses
    pose = [5*sin(t + pi/4), 5*cos(t + pi/4), 5, 0.1*sin(t), 0.2*cos(t), 0.3*sin(t)];  % Example pose
end

function int_mod = modPosNeg(val)
    int_mod = val;
    for i = 1:length(val)
        if val(i) >= 0
            int_mod(i) = mod(val(i),4096);
        else
            int_mod(i) = mod(val(i),-4096);
        end
    end
end


%% Event handler
function eventhandlerXsens(varargin)
    global h IMUData
    % only action when new datapacket arrived
    dataPacket = varargin{3}{2};

    if h.XsDataPacket_containsOrientation(dataPacket)
        IMUData{1} = h.XsDataPacket_packetCounter(dataPacket);
        IMUData{2} =  h.XsDataPacket_sampleTimeFine(dataPacket);
        IMUData{3} = cell2mat(h.XsDataPacket_orientationQuaternion(dataPacket,h.XsDataIdentifier_XDI_CoordSysEnu));
        % fprintf('\r[%d,%d]: q0: %.2f, q1: %.2f, q2: %.2f, q3: %.2f',packetcounter,timestp, quaternion(1), quaternion(2), quaternion(3), quaternion(4));
    end
    h.dataPacketHandled(varargin{3}{1}, dataPacket);
end
