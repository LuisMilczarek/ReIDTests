#!/usr/bin/env python
import torchreid


def main() -> None:
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=['market1501', 'dukemtmcreid','cuhk03'],
        height=256,
        width=128,
        batch_size_train=256,
        transforms=['random_flip','random_crop']
    )
    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    model = model.cuda()

    optmizer  = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optmizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optmizer,
        scheduler=scheduler,
        label_smooth=True
    )
    start_epoch = torchreid.utils.resume_from_checkpoint(
        'log/resnet50/model/model.pth.tar-30',
        model,
        optmizer
    )


    engine.run(
        save_dir="log/resnet50",
        max_epoch=120,
        eval_freq=10,
        print_freq=10,
        test_only=False,
        start_epoch=start_epoch
    )

if __name__ == "__main__":
    main()