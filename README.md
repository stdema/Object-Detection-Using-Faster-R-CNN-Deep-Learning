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
