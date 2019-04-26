def run():

    # real code goes here
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Download the pyTorch dataset into the folder DataSet in current working directory.
    # The DataSet contains the ten category of images.
    # Plane, Car, Bird, Cat, Deer,Dog, Frog, Horse, Ship, Truck
    # Cifar10 dataset is available directly from torchvision, is popular dataset for Image Clasification
    # train = true gives the training data set.
    # transformt the image into tensor.
    trainset= torchvision.datasets.CIFAR10(root='./DataSet',
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
    # Total 50 thousand images.
    # print(trainset)

    # Feed in your trainigng dataset to neural network in batches.
    # Pytorch makes it easy by using below function where can specify configuration.
    # each bach will include 8 image.
    # its common practice to Shuffle the data when working on machine learning, Prevents model from picking arbitary pattern
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=8, # default is 1
                                            shuffle=True, 
                                            num_workers=2)
    # Laod the test dataset
    # train= false ..gives the test data.
    testset= torchvision.datasets.CIFAR10(root='./DataSet',
                                        train=False,
                                        download=True,
                                        transform=transforms.ToTensor())

    # Total 10 thousand test images.
    # print(testset) 

    # we do not need the test data to be randomized so setting shuffle=False
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=8, # default is 1
                                            shuffle=False, 
                                            num_workers=2) 

    # 10 labels corresponding to ten category.
    alllabels = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # below code is just for showing the some of the training images.
    import matplotlib.pyplot as plt
    import numpy as np     

    images_batch, labels_batch= iter(trainloader).next()

    # images_batch what we are going to feed to the neural network.
    # batch Size-8
    # channels(RGB)-3 color image.
    # each image is 32*32 pixesl.
    print(images_batch.shape)
   
    # Using torchvision utility to make a grid of images in this batch.
    # make_grid function will place image side by side
    #Place 8 image in row with 2pixel padding applied. height becomes 36 and width 274
    img = torchvision.utils.make_grid(images_batch)

    # print(img.shape)
    #np.transpose(img, (1,2,0)).shape
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.axis('off')
    plt.show()

    # Constructing the Convultional Neural Network for Image Classification
    # Use the Torch.nn library for setting up the layer.
    import torch.nn as nn

    in_size = 3 # input is the 3 channel of image
    hid1_size = 16 # number of channels output by first convolutional layer . 16 feature maps.  
    hid2_size = 32 # 2nd layer will output 32 feature maps
    out_size = len (alllabels) # length 10 as total category is 10
    k_conv_size = 5  # 5*5  convolutional kernel
    #gradient vanishing/ Exploding Solution.(Normalize) Subtract mean and divide by standard deviation.
    # (Scale)multiply by constant 
    # (Shift) add constant

    class ConvNet(nn.Module):

        def __init__(self):
            super(ConvNet,self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_size,hid1_size,k_conv_size),
                nn.BatchNorm2d(hid1_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))

            self.layer2 = nn.Sequential(
                nn.Conv2d(hid1_size,hid2_size,k_conv_size),
                nn.BatchNorm2d(hid2_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))

            self.fc = nn.Linear(hid2_size * k_conv_size * k_conv_size,out_size)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out. reshape(out.size(0),-1)
            out = self.fc(out)

            return out
    
    # Inoput data becomes more deeper because of Convolution and smaller after pooling.

    # Load existing saved model if there is any.
    # Comment this section while training.
    # Start of code
    saved_model=ConvNet()
    saved_model.load_state_dict(torch.load('trained_model.pth'))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = saved_model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct +=(predicted ==labels).sum().item()
  
    print('Accuracy of the saved model on the 10000 test images: {}%'.format(100 * correct /total))

    # verify sample data
    print('Test Data: ', ' '.join('%5s' % alllabels[labels_batch[j]] for j in range(8)))
    test_output = saved_model(images_batch)
    _, predicted_test = torch.max(test_output.data,1)

    print('Prediction: ', ' '.join('%5s' % alllabels[predicted_test[j]] for j in range(8)))

    # End of Code.

    # instantiate and train our Convolutional Neural Network.
    # instantiaite Learning rate as 0.001
    # loss function as CrossEntropyLoss
    # we are using adam optimizer to calculate the gradient descent.

    model = ConvNet()
    learnig_rate = 0.001
    criterion = nn.CrossEntropyLoss() # loss function used for 
    optimizer= torch.optim.Adam(model.parameters(),
                                lr=learnig_rate) 


    total_step = len(trainloader)
    num_epochs = 5

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            outputs = model(images)
            loss = criterion(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # updtates the model parameters.

            if (i+1) % 2000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs,i+1,total_step,loss.item()))
   
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct +=(predicted ==labels).sum().item()
  
    print('Accuracy of the model on the 10000 test images: {}%'.format(100 * correct /total))
    #torch.save(model.state_dict(),'trained_model.pth') --Commented
   

if __name__ == '__main__':  
    run()

