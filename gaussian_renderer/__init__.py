from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
# from gaussian_renderer.neilf_RF import render_neilf_RF
from gaussian_renderer.RF_render import render_RF
# from gaussian_renderer.RF_render_calibration import render_neilf_RF_calibration
# from gaussian_renderer.RF_render_RSSI_prediction import render_neilf_RSSI_prediction
# from gaussian_renderer.RF_render_RSSI_prediction_complex import render_neilf_RSSI_prediction_complex
from gaussian_renderer.RF_render_wideband import render_RF_bb
from gaussian_renderer.RF_render_RSSI_prediction_LoSNLoS import render_RF_RSSI_prediction


render_fn_dict = {
    "render": render,           
    "render_RF": render_RF,
    "render_RF_bb": render_RF_bb,
    "render_RF_RSSI_prediction": render_RF_RSSI_prediction
}




