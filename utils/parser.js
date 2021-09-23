
const createParameter = (args, data) => {
    const moduleArgsParser = {
        "modelSelect":"--model",
        "datasetSelect": "--dataset",
        "augmentationSelect": "--augmentation",
        "criterionSelect": "--criterion",
        "optimizerSelect": "--optimizer",
        "batchSize": "--batch_size",
        "epochs":"--epochs",
        "valRatio": "--val_ratio",
        "learningRate": "--lr",
        "lrDecayStep": "--lr_decay_step",
        "logInterval": "--log_interval",
        "seed": "--seed",
    }
    for (key in data){
        option = moduleArgsParser[key];
        value = data[key];
        //value = syncDataType(option, data[key]);

        args.push(option);
        args.push(value);
    }
    return args;
};

const syncDataType = (key, value) => {
    const moduleArgsType = {
        "--model":"str",
        "--dataset":"str",
        "--augmentation":"str",
        "--criterion":"str",
        "--optimizer":"str",
        "--batch_size":"int",
        "--epochs":"int",
        "--val_ratio":"float",
        "--lr":"float",
        "--lr_decay_step":"int",
        "--log_interval":"int",
        "--seed":"int",
    };

    if (moduleArgsType[key] == "str"){
        return "'" + value + "'";
    };

    return value;
}
module.exports = {
    createParameter,
};