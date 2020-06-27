
class Classification():
    '''
    accumulates training status for classification models
    
    arg
        best_loss_init: int (default: 1.e10)
            initual best loss
    '''
    def __init__(self,
        best_loss_init=1.e10
    ):
        self.status = {
            'train' : {
                'loss' : [],
                'acc'  : []
            },
            'val'   : {
                'loss' : [],
                'acc'  : []
            }
        }
        self.best_epoch = 0
        self.best_loss = best_loss_init
        self.__save_flag = False
    
    def append_train(self,
        loss, accuracy
    ):
        '''
        append training loss and accuracy

        arg
            loss: int
                training loss for an epoch
            accuracy: int
                training accuracy for an epoch
        '''
        self.status['train']['loss'].append(loss)
        self.status['train']['acc'].append(accuracy)
    
    def append_validation(self,
        loss, accuracy
    ):
        '''
        append validation loss and accuracy

        arg
            loss: int
                validation loss for an epoch
            accuracy: int
                validation accuracy for an epoch
        '''
        self.status['val']['loss'].append(loss)
        self.status['val']['acc'].append(accuracy)

        if self.best_loss > loss:
            self.best_epoch = len(self.status['val']['loss'])
            self.best_loss  = loss
            self.__save_flag = True
    
    def verbose(self,
        epochs, epoch=None, sec=None
    ):
        '''
        verbose

        arg
            epochs: int
                total training epochs
            epoch: int (default: None)
                current epoch count.
                if None, the length of appended status will be used
            sec: float (default: None)
                time training an epoch
                if None, the training time will not be printed
        '''
        epoch = epoch if epoch else len(self.status['train']['loss'])
        msg = ''

        if sec:
            msg += '[{:7.1f} sec]\t'.format(sec)
        msg += '[{:4} / {:4}]\t'.format(epoch, epochs)
        msg += '[train LOSS : {:.5f}, ACC : {:.5f}]\t'.format(self.status['train']['loss'][-1], self.status['train']['acc'][-1])
        msg += '[val LOSS : {:.5f}, ACC : {:.5f}]'.format(self.status['val']['loss'][-1], self.status['val']['acc'][-1])

        print(msg)

    def should_save(self):
        '''
        returns True if the best loss is updated
        '''
        if self.__save_flag:
            self.__save_flag = False
            return True
        else:
            return False

    def plot(self,
        filename='./curve.png'
    ):
        '''
        plots the status

        arg
            filename: str (default: ./curve.png)
                filename for output picture
        '''

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        train_loss = self.status['train']['loss']
        train_acc  = self.status['train']['acc']
        val_loss   = self.status['val']['loss']
        val_acc    = self.status['val']['acc']

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.title('Model Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.tight_layout()

        plt.savefig(filename)
        
        plt.close()

class Regression():
    '''
    accumulates training status for regression models
    
    arg
        best_loss_init: int (default: 1.e10)
            initual best loss
    '''
    def __init__(self,
        best_loss_init=1.e10
    ):
        self.status = {
            'train' : [],
            'val'   : []
        }
        self.best_epoch = 0
        self.best_loss = best_loss_init
        self.__save_flag = False

    def append_train(self,
        loss
    ):
        '''
        append train loss

        arg
            loss: int
                traning loss for an epoch
        '''
        self.status['train'].append(loss)
    
    def append_validation(self,
        loss
    ):
        '''
        append validation loss

        arg
            loss: int
                validation loss for an epoch
        '''
        self.status['val'].append(loss)

        if self.best_loss > loss:
            self.best_loss = loss
            self.best_epoch = len(self.status['val'])
            self.__save_flag = True

    def verbose(self,
        epochs, epoch=None, sec=None
    ):
        '''
        verbose

        arg
            epochs: int
                total training epochs
            epoch: int (default: None)
                current epoch count.
                if None, the length of appended status will be used
            sec: float (default: None)
                time training an epoch
                if None, the training time will not be printed
        '''

        epoch = epoch if epoch else len(self.status['train'])
        msg = ''

        if sec:
            msg += '[{:7.1f} sec]\t'.format(sec)
        msg += '[{:4} / {:4}]\t'.format(epoch, epochs)
        msg += '[train LOSS : {:.5f}]\t'.format(self.status['train'][-1])
        msg += '[val LOSS : {:.5f}]\t'.format(self.status['val'][-1])

        print(msg)

    def should_save(self):
        '''
        returns True if the best loss is updated
        '''
        if self.__save_flag:
            self.__save_flag = False
            return True
        else:
            return False

    def plot(self,
        filename='./curve,png'
    ):
        '''
        plots the status

        arg
            filename: str (default: ./curve.png)
                filename for output picture
        '''

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        train_loss = self.status['train']
        val_loss   = self.status['val']

        plt.figure(figsize=(12, 8))

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.tight_layout()

        plt.savefig(filename)
        
        plt.close()

class GAN():
    '''
    accumulates training status for GANs
    '''
    def __init__(self):
        self.status = {
            'G' : [],
            'D' : []
        }
    
    def append(self,
        G_loss, D_loss
    ):
        '''
        append generator and discriminator loss

        arg
            G_loss: int
                generator loss
            D_loss: int
                discriminator loss
        '''

        self.status['G'].append(G_loss)
        self.status['D'].append(D_loss)

    def verbose(self,
        batches_done,
        sec=None
    ):
        '''
        verbose

        arg
            batchesw_done: int
                batches done
            sec: float (default: None)
                time training an epoch
                if None, the training time will not be printed
        '''
        msg = ''

        if sec:
            msg += '[{:7.1f} sec]\t'.format(sec)
        msg += '[{:8}]\t'.format(batches_done)
        msg += '[G LOSS : {:.5f}]\t'.format(self.status['G'][-1])
        msg += '[D LOSS : {:.5f}]'.format(self.status['D'][-1])

        print(msg)

    def plot(self,
        filename='./curve.png'
    ):
        '''
        plots the status

        arg
            filename: str (default: ./curve.png)
                filename for output picture
        '''

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        G_loss = self.status['G']
        D_loss = self.status['D']

        plt.figure(figsize=(12, 8))

        plt.plot(G_loss)
        plt.plot(D_loss)
        plt.title('Model Loss')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.legend(['Generator', 'Discriminator'], loc='upper left')

        plt.tight_layout()

        plt.savefig(filename)
        
        plt.close()