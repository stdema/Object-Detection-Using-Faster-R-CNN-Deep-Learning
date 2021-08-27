# Object-Detection-Using-Faster-R-CNN-Deep-Learning
 Faster R-CNN(Regions with Convolutional Neural Networks) 사물 검출기를 훈련시키는 방법을 다룹니다.

### 사전 훈련된 검출기 다운로드하기
훈련이 완료될 때까지 기다릴 필요가 없도록 사전 훈련된 검출기를 다운로드합니다.
```c
doTrainingAndEval = false;
if ~doTrainingAndEval && ~exist('fasterRCNNResNet50EndToEndVehicleExample.mat','file')
    disp('Downloading pretrained detector (118 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/fasterRCNNResNet50EndToEndVehicleExample.mat';
    websave('fasterRCNNResNet50EndToEndVehicleExample.mat',pretrainedURL);
end
```
### 데이터 세트 불러오기
295개의 영상을 포함하는 규모가 작은 레이블 지정된 데이터셋을 사용합니다.
각 영상에는 차량에 대해 레이블 지정된 건수가 한 건 또는 두 건 있습니다.
```c
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;
```
첫 번째 열은 영상 파일 경로를 포함하고, 두 번째 열은 차량 경계 상자를 포함합니다.

데이터셋을 훈련 세트, 검증 세트, 테스트 세트로 분할합니다.
60%를 훈련용으로, 10%를 검증용으로, 나머지를 훈련된 검출기의 테스트용으로 선택합니다.
```c
rng(0)
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * height(vehicleDataset));

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);
```
imageDatastore와 boxLabelDatastore를 사용하여 훈련과 평가 과정에서 영상 및 레이블 데이터를 불러오기 위한 데이터저장소를 만듭니다.
```c
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));
```
영상 데이터저장소와 상자 레이블 데이터저장소를 결합합니다.
```c
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
```
상자 레이블과 함께 훈련 영상 중 하나를 표시합니다.
```c
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
```
![화면 캡처 2021-08-28 045011](https://user-images.githubusercontent.com/86040099/131181944-429e8661-3056-4208-be83-67edcc2cc5fa.png)

###Faster R-CNN 검출 신경망 만들기
Faster R-CNN 객체 검출 신경망은 하나의 특징 추출 신경망과 그 뒤에 오는 2개의 하위 신경망으로 구성됩니다.
특징 추출 신경망은 일반적으로 ResNet-50, Inception v3과 같은 사전 훈련된 CNN입니다.
특징 추출 신경망 뒤에 오는 첫 번째 하위 신경망은 사물 제안(영상에서 사물이 존재할 가능성이 있는 영역)을 생성하도록 훈련된 영역 제안 신경망(RPN)입니다.
두 번째 하위 신경망은 각 사물 제안의 실제 클래스를 예측하도록 훈련됩니다.
특징 추출에 ResNet-50을 사용합니다.
응용 요구 사항에 따라 MobileNet v2나 ResNet-18과 같은 여타 사전 훈련된 신경망도 사용할 수 있습니다.

fasterRCNNLayers를 사용하여, 사전 훈련된 특징 추출 신경망이 주어졌을 때 자동으로 Faster R-CNN 신경망을 만듭니다.
먼저 신경망 입력 크기를 지정합니다. 
신경망 입력 크기를 선택할 때는 신경망 자체를 실행하는 데 필요한 최소 크기, 훈련 영상의 크기, 그리고 선택한 크기에서 데이터를 처리할 때 발생하는 계산 비용을 고려해야 합니다. 
소요되는 계산 비용을 줄이기 위해 신경망을 실행하는 데 필요한 최소 크기인 [224 224 3]으로 신경망 입력 크기를 지정하십시오.
```c
inputSize = [224 224 3];
```
다음으로, estimateAnchorBoxes를 사용하여 훈련 데이터의 사물 크기를 기반으로 앵커 상자를 추정합니다. 
transform을 사용하여 훈련 데이터를 전처리한 후에 앵커 상자의 개수를 정의하고 앵커 상자를 추정합니다.
```c
preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)
```
이제 resnet50을 사용하여 사전 훈련된 ResNet-50 모델을 불러옵니다.
```c
featureExtractionNetwork = resnet50;
```
'activation_40_relu'를 특징 추출 계층으로 선택합니다.
```c
featureLayer = 'activation_40_relu';
```
검출할 클래스의 개수를 정의합니다.
```c
numClasses = width(vehicleDataset)-1;
```
Faster R-CNN 객체 검출 신경망을 만듭니다.
```c
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
```

###데이터 증대
transform을 사용하여 영상과 영상에 해당하는 상자 레이블을 가로 방향으로 무작위로 뒤집어서 훈련 데이터를 증대합니다.
동일한 영상을 여러 차례 읽어 들이고 증대된 훈련 데이터를 표시합니다.
```c
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)
```

###훈련 데이터 전처리하기
증대된 훈련 데이터와 검증 데이터를 전처리하여 훈련에 사용할 수 있도록 준비합니다.
```c
trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));
```
영상과 경계 상자를 표시합니다.
```c
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
```
![화면 캡처 2021-08-28 054743](https://user-images.githubusercontent.com/86040099/131186894-40c4fe3a-fd92-4c4b-9e7d-1e5b2f58c985.png)

###Faster R-CNN 훈련시키기
trainingOptions를 사용하여 신경망 훈련 옵션을 지정합니다.
'ValidationData'를 전처리된 검증 데이터로 설정합니다.
'CheckpointPath'를 임시 위치로 설정합니다.
```c
options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);
```
doTrainingAndEval이 true인 경우, trainFasterRCNNObjectDetector를 사용하여 Faster R-CNN 사물 검출기를 훈련시킵니다. 그렇지 않은 경우는 사전 훈련된 신경망을 불러오십시오.
```c
if doTrainingAndEval
    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
else
    % Load pretrained detector for the example.
    pretrained = load('fasterRCNNResNet50EndToEndVehicleExample.mat');
    detector = pretrained.detector;
end
```
신경망을 훈련시키는 데는 약 20분정도가 소요됩니다.
짧게 확인해 보려면 하나의 테스트 영상에 대해 검출기를 실행하십시오.
```c
I = imread(testDataTbl.imageFilename{1});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
```
결과를 표시합니다.
```c
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
```
![화면 캡처 2021-08-28 055241](https://user-images.githubusercontent.com/86040099/131187336-e1b4df2d-7233-435b-ad6f-aa94e0036c48.png)

