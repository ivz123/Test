#include <stdio.h>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr);

// Raw features copied from test sample (Edge Impulse > Model testing)
static float input_buf[] = {
    /* Paste your raw features here! */ 
    4.7631, 7.8243, 0.0513, 5.5468, 8.7637, -0.5525, 5.8438, 9.5105, 0.0455, 4.8712, 9.3263, 0.2258, 4.5038, 9.2543, 0.4806, 4.5140, 9.2524, 0.6472, 4.6669, 8.4472, 0.5691, 4.4172, 8.3052, 0.4437, 4.2184, 8.5835, 0.1888, 4.6383, 8.8400, -0.1847, 4.8004, 8.6046, 0.1148, 5.5701, 8.4925, -0.4860, 6.3244, 8.0507, -0.8959, 6.3659, 7.5621, -0.6892, 5.8896, 7.3061, -0.7260, 5.4784, 7.6128, -0.6120, 5.4476, 7.8645, -0.7977, 5.3063, 8.1237, -0.7497, 5.7233, 7.9748, -0.6846, 5.7480, 8.0308, -0.2099, 5.4913, 8.0622, -0.1028, 5.9243, 8.6140, -0.1110, 6.1054, 8.8840, 0.1287, 6.2471, 9.3643, 0.4045, 6.0211, 10.2348, 0.8033, 5.5679, 10.6719, 0.9875, 4.9725, 11.5824, 0.9402, 5.0521, 12.1590, 0.7320, 4.8662, 12.5877, 0.0875, 6.2317, 13.0896, 0.4802, 6.3445, 12.9592, 0.9055, 6.5680, 12.0092, 1.2721, 5.8125, 10.8766, 0.3223, 5.0639, 10.8101, -0.5608, 2.8346, 9.7475, -4.1128, 4.3090, 11.7796, -4.1595, 5.1571, 11.4274, -2.1356, 4.2714, 10.4486, -1.4760, 3.4792, 9.5811, -1.2870, 3.6966, 8.9786, -1.5004, 4.8896, 10.1387, -1.9585, 4.6756, 10.2535, -2.5513, 1.6384, 9.2424, -3.1270, 1.0051, 5.3855, -0.5926, 1.9487, 4.7461, -0.0943, 2.7605, 4.9377, -2.5578, 2.4015, 5.4110, -2.7532, 2.3939, 5.5692, -2.7649, 1.6252, 4.8314, -2.4542, 0.8663, 4.8834, -3.2377, 0.9466, 4.8330, -3.6479, 1.1203, 3.9561, -3.1469, 1.0159, 3.8072, -2.6423, 1.8006, 3.8782, -4.2309, 2.6861, 4.8677, -5.2759, 2.8983, 5.2713, -5.9390, 2.5723, 6.5499, -3.8518, 2.5262, 7.7698, -5.7444, 2.4530, 8.5941, -5.4574, 3.3038, 9.7196, -5.5258, 2.7185, 11.0225, -6.4447, 2.9528, 11.8447, -6.6527, 5.2720, 13.3166, -6.3394, 6.2906, 11.0989, -6.2529, 5.7470, 10.4335, -6.3262, 4.3735, 9.2551, -3.5786, 1.5344, 7.1343, -1.4599, 0.4509, 6.6399, -1.8115, 0.1535, 7.1949, -0.6198, 2.2926, 9.5981, 0.6207, 3.7784, 11.5284, 0.2839, 5.7380, 12.0325, -1.0029, 5.4684, 10.0151, -1.5669, 4.8619, 9.1494, -1.6979, 3.8446, 7.8881, -0.6485, 3.0345, 7.9755, 0.4477, 2.7987, 8.5172, 0.3265, 3.2561, 9.5346, 0.4331, 4.1843, 9.3978, 0.3903, 4.6862, 8.9340, 0.3138, 4.4477, 8.3945, -0.1209, 3.2159, 7.7568, -0.5709, 3.3757, 7.2666, -0.9300, 3.8294, 7.1385, 0.0207, 3.9018, 6.8552, -0.1184, 3.6622, 6.5246, -0.0564, 3.8301, 5.9245, -0.1052, 3.8988, 6.2100, -0.1465, 4.2671, 6.3401, -0.0780, 4.6117, 7.4163, -0.2652, 4.3479, 8.1370, -0.3344, 4.4402, 8.4524, 0.4121, 4.9933, 9.7776, 0.4826, 5.1172, 10.4701, 0.4046, 5.5098, 12.3491, 0.9218, 6.1187, 13.3513, 0.3769, 5.9744, 13.1449, 0.4203, 5.8003, 13.6637, 0.4554, 5.1592, 14.1244, 1.7262, 4.9526, 14.9475, 1.6946, 2.8838, 13.5239, 0.9219, 2.3825, 12.6985, -0.5914, 2.5723, 12.2906, -0.7997, 3.5014, 10.5833, -2.1347, 4.5619, 10.1655, -1.2658, 5.1876, 9.4405, -0.5695, 4.5729, 10.3207, -2.2027, 3.5087, 12.3779, -2.5334, 3.1864, 11.7724, -2.9281, 3.7661, 9.8915, -2.4101, 3.6693, 9.0739, -0.9888, 2.7687, 8.7326, -1.0244, 1.1881, 8.4069, -2.4934, 0.6404, 6.9062, -0.8112, 1.6900, 5.6391, -0.3925, 2.3029, 4.2877, -1.1226, 1.5202, 5.2744, -2.2040, 1.2347, 5.7573, -2.8171, 1.4805, 5.5695, -3.3082, 1.0042, 4.6186, -2.2191, 1.1681, 3.9172, -2.2247, 0.4269, 3.5799, -3.5756, 0.9753, 3.4870, -3.4779, 0.2041, 3.7795, -3.4815, 0.9230, 4.1379, -4.0363
};

int main(int argc, char **argv) {
    
    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

    // Calculate the length of the buffer
    size_t buf_len = sizeof(input_buf) / sizeof(input_buf[0]);

    // Make sure that the length of the buffer matches expected input length
    if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printf("ERROR: The size of the input buffer is not correct.\r\n");
        printf("Expected %d items, but got %d\r\n", 
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, 
                (int)buf_len);
        return 1;
    }

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;

    // Perform DSP pre-processing and inference
    res = run_classifier(&signal, &result, false);

    // Print return code and how long it took to perform inference
    printf("run_classifier returned: %d\r\n", res);
    printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n", 
            result.timing.dsp, 
            result.timing.classification, 
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n", 
                bb.label, 
                bb.value, 
                bb.x, 
                bb.y, 
                bb.width, 
                bb.height);
    }

    // Print the prediction results (classification)
#else
    printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        printf("  %s: ", ei_classifier_inferencing_categories[i]);
        printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

    return 0;
}

// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (input_buf + offset)[i];
    }

    return EIDSP_OK;
}