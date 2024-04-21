def eval_model(model, test_loader, criterion, optimizer, num_epochs=10):

        # Validation phase
        model.eval()  # Set model to evaluate mode
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.cuda(), labels.cuda()
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Val Loss: {total_val_loss / len(test_loader)}, Accuracy: {correct / len(test_loader.dataset)}')
