%Đọc dữ liệu và chia tập train/val
imds = imageDatastore('E:\Data_training', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized'); % 80% train, 20% validation
%Thiết kế mạng CNN:

inputSize = [64 64 3];% Resize ảnh về 64x64 (nhẹ, nhanh)

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 8, 'Padding', 'same')  % 8 bộ lọc 3x3
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(4)      % 4 lớp tương ứng 4 loại ảnh
    softmaxLayer
    classificationLayer];
% Training:
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsVal = augmentedImageDatastore(inputSize, imdsVal);

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain, layers, options);
%Kiểm tra kết quả:
YPred = classify(net, augimdsVal);
YTrue = imdsVal.Labels;

accuracy = sum(YPred == YTrue) / numel(YTrue);
disp(['Độ chính xác: ', num2str(accuracy * 100), '%']);

confusionchart(YTrue, YPred);  % Cho bạn coi mô hình nhầm chỗ nào
%Dự đoán ảnh mới:
imageFolder = 'E:\Data_testAI';
imageFiles = dir(fullfile(imageFolder, '*.jpg'));
numImages = length(imageFiles);
for i = 1:length(imageFiles)
filePath = fullfile(imageFolder, imageFiles(i).name);
img = imread(filePath);
img = imresize(img, inputSize(1:2));
label = classify(net, img);
   subplot(ceil(sqrt(numImages)), ceil(sqrt(numImages)), i);
    imshow(img);
    title(['dự đoán: ', char(label)]);
end