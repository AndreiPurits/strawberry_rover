## FPS dataset check

- **dataset**: `/home/andrei/project/strawberry_rover_ws/data/ФПС ДАТАСЕТ`
- **status**: **ok**

### Structure
- **images**: `/home/andrei/project/strawberry_rover_ws/data/ФПС ДАТАСЕТ/images` (exists=1)
- **labels**: `/home/andrei/project/strawberry_rover_ws/data/ФПС ДАТАСЕТ/labels` (exists=1)
- **images_with_labels**: `/home/andrei/project/strawberry_rover_ws/data/ФПС ДАТАСЕТ/images_with_labels` (exists=1)

### Counts
- **images**: 600
- **labels**: 600
- **images_with_labels**: 600

### Name matching
- **missing labels for images**: 0
- **missing images for labels**: 0

### Images
- **broken images**: 0

### Labels (YOLO-seg)
- **empty label files**: 9
- **class ids present**: `[0, 1, 2, 3]`
- **parse errors**: 0
- **lines**: total=1273 with_poly=1273 bbox_only=0

### Sample label files (first 3 non-empty)
- `test__1003_png.rf.a3141b75dfbe219d4becbdc5b0549d8d.txt`

```
1 0.638281 0.591406 0.107813 0.220312 0.614062 0.482812 0.585938 0.525000 0.585938 0.579688 0.614062 0.684375 0.643750 0.704688 0.664062 0.670312 0.693750 0.543750 0.684375 0.517188 0.653125 0.482812
2 0.703125 0.011719 0.031250 0.023438 0.687500 0.000000 0.718750 0.000000 0.718750 0.023438 0.687500 0.023438
```

- `test__100_png.rf.38cc6463404857cebc29278deeecce8f.txt`

```
1 0.503906 0.352344 0.064062 0.129688 0.471875 0.287500 0.535937 0.287500 0.535937 0.417187 0.471875 0.417187
```

- `test__1012_png.rf.cb150ec09c303985eaf975c8864247cf.txt`

```
1 0.627344 0.670312 0.201563 0.365625 0.659375 0.490625 0.634375 0.490625 0.629687 0.517188 0.596875 0.534375 0.587500 0.504687 0.532813 0.546875 0.532813 0.664062 0.579688 0.817187 0.604688 0.853125 0.623437 0.851562 0.651563 0.815625 0.726562 0.629687 0.715625 0.560937
1 0.299219 0.435156 0.185938 0.351562 0.268750 0.262500 0.215625 0.335938 0.215625 0.431250 0.254688 0.589063 0.287500 0.612500 0.321875 0.578125 0.378125 0.459375 0.384375 0.373437 0.367188 0.323437 0.346875 0.323437 0.300000 0.262500
```

