import time

import torch


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model, args):
    args = args or {}
    name = (args.get("optimizer") or "sgd").lower()
    lr = args.get("lr", 0.01)
    weight_decay = args.get("weight_decay", 5e-4)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.get("momentum", 0.9),
        weight_decay=weight_decay,
        nesterov=args.get("nesterov", True),
    )


class CustomModelTrainer:
    def __init__(self) -> None:
        pass

    def train_model(
        self,
        model,
        results,
        train_loader,
        epochs,
        timeout_s: float = None,
        loss_func=None,
        optimizer=None,
        device=torch.device("cpu"),
        test_loader=None,
        args: dict = None,
        start_time: float = None,
    ):
        args = args or {}
        device = _model_device(model)
        model.to(device)
        model.train()

        if loss_func is None:
            loss_func = torch.nn.CrossEntropyLoss
        cost = loss_func()

        if optimizer is None:
            optimizer = _build_optimizer(model, args)
        else:
            optimizer.param_groups.clear()
            optimizer.state.clear()
            optimizer.add_param_group({"params": list(model.parameters())})

        start_time = start_time or time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        total_mini_batches = 0
        float_epochs = 0.0
        data_entries = max(len(train_loader), 1)

        try:
            for epoch in range(epochs):
                num_mini_batches = 0
                for train_x, train_label in train_loader:
                    if timeout_s and (time.time() - start_time) > timeout_s:
                        break

                    train_x = train_x.to(device, non_blocking=True)
                    train_label = train_label.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    predict_y = model(train_x)
                    loss = cost(predict_y, train_label)
                    loss.backward()
                    optimizer.step()

                    batch_n = train_label.size(0)
                    total += batch_n
                    total_loss += loss.item() * batch_n
                    correct += (predict_y.argmax(1) == train_label).sum().item()
                    total_mini_batches += 1
                    num_mini_batches += 1
                    float_epochs = epoch + (num_mini_batches / data_entries)

                if timeout_s and (time.time() - start_time) > timeout_s:
                    break
        except Exception as e:
            print(f"CustomModelTrainer.train_model exception: {e}")

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)

        return {
            "time_taken_s": time.time() - start_time,
            "num_epochs": float_epochs,
            "total_mini_batches": total_mini_batches,
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def validate_model(
        self,
        model,
        dataloader,
        device: str = "cpu",
        loss_func=None,
        optimizer=None,
        round_no=None,
        args: dict = None,
    ):
        if loss_func is None:
            loss_func = torch.nn.CrossEntropyLoss

        if isinstance(device, str):
            device = torch.device(device)
        device = _model_device(model) if device.type == "cpu" else device

        model.eval()
        model.to(device)

        cost = loss_func()
        total_loss = 0.0
        correct = 0
        count = 0
        batches = 0

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                y_pred = model(x_batch)
                loss = cost(y_pred, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                correct += (y_pred.argmax(1) == y_batch).sum().item()
                count += y_batch.size(0)
                batches = i + 1

        model.train()
        res = {
            "accuracy": 100.0 * correct / max(count, 1),
            "loss": total_loss / max(count, 1),
        }
        print(f"CustomModelTrainer.validate_model round={round_no} res={res}")
        return res
