def get_outputs_feats(model, data, args):
    if args.classifier_type == "A":
        spectrogram, labels = data
        spectrogram = spectrogram.to(args.device)
        labels = labels.to(args.device)
        outputs, feats = model(spectrogram)

    elif args.classifier_type == "V":
        frames, labels = data
        frames = frames.to(args.device)
        labels = labels.to(args.device)
        outputs, feats = model(frames)
    else:
        # need to implement
        pass
    return outputs, feats
