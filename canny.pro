TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

HEADERS += \
    include/opencv2/aruco.h \
    include/opencv2/calib3d/calib3d.hpp \
    include/opencv2/core/core.hpp \
    include/opencv2/features2d/features2d.hpp \
    include/opencv2/highgui.hpp \
    include/opencv2/highgui/highgui.hpp \
    include/opencv2/highgui/highgui_c.h \
    include/opencv2/imgproc.hpp \
    include/opencv2/imgproc/imgproc.hpp \
    include/opencv2/imgproc/imgproc_c.h \
    include/opencv2/imgproc/types_c.h \
    include/opencv2/opencv.hpp

INCLUDEPATH += $$PWD/.
DEPENDPATH += $$PWD/.

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/lib/ -lopencv_world3410
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/lib/ -lopencv_world3410d

INCLUDEPATH += $$PWD/include
DEPENDPATH += $$PWD/include


