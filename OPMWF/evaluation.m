clear all;
num1=num2str(2);%source

% 假設 groundTruth 和 transformMatrixB 是已經定義好的 4x4 矩陣
groundTruth = [    
0.9990999988 0.04241671735 0.0001256093891 -5.270430071
-0.04241652512 0.999099217 -0.001261660767 5.894057019
-0.0001790117498 0.001255197356 0.9999991966 -0.4711202578
0 0 0 1



]; % 替換成實際的 ground truth 矩陣


transformMatrixB = [ 
0.999107022	0.0422508943	0.0001432652	-5.2736329
-0.0422507324	0.9991065495	-0.0009891365	5.896345304
-0.0001849291	0.0009822002	0.9999995005	-0.4736542478
0	0	0	1
]; % 替換成實際的變換矩陣 B

%transformMatrixB = calinv(transformMatrixB);

% 分解矩陣以獲取旋轉和平移部分
R_groundTruth = groundTruth(1:3, 1:3);
T_groundTruth = groundTruth(1:3, 4);

R_transformB = transformMatrixB(1:3, 1:3);
T_transformB = transformMatrixB(1:3, 4);

% 計算旋轉誤差
R_error = R_groundTruth' * R_transformB; % 相對旋轉

xxx=(trace(R_error) - 1);
angle_error = acos((trace(R_error) - 1) / 2); % 轉換為角度誤差
rotation_error = rad2deg(angle_error); % 轉換為度

% 計算平移誤差
translation_error = norm(T_groundTruth - T_transformB);
source_name=['keypoint\keypoint',num1,'.pcd'];
targert_name=['keypoint\keypoint',num2,'.pcd'];
disp(source_name);
source=pcread(source_name);
target=pcread(targert_name);
SP = double(source.Location');
P1 = transformMatrixB(1:3,1:3)*SP+repmat(transformMatrixB(1:3,4),1,size(SP,2));
P2 = groundTruth(1:3,1:3)*SP+repmat(groundTruth(1:3,4),1,size(SP,2));
rmse = sqrt(sum(sum((P1-P2).^2))/size(SP,2));


% 顯示結果
fprintf('Rotation Error (in degrees): %f\n', rotation_error);
fprintf('Translation Error: %f\n', translation_error);
fprintf('RMSE: %f\n', rmse);

function T_inv = calinv(T)
    % Extract the rotation matrix and translation vector from the transformation matrix
    R = T(1:3, 1:3);
    t = T(1:3, 4);

    % Compute the transpose of the rotation matrix (which is the inverse of a rotation matrix)
    R_inv = R';

    % Compute the new translation vector
    t_inv = -R_inv * t;

    % Combine them into the inverse transformation matrix
    T_inv = [R_inv, t_inv; 0 0 0 1];

    % Return the inverse transformation matrix
end

