from mmdet.core import bbox2result, bbox2result_add_kps
from ..builder import DETECTORS, build_detector
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn


@DETECTORS.register_module()
class SCRFD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 frozenall_skip_layers=None,
                 frozen_layers=None,
                 reset_head_weight=False,
                 frozen_head_levels=None,
                 to_onnx_flag=False,

                 kd_teacher_model = None
                 ):
        super(SCRFD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

        self.to_onnx_flag = to_onnx_flag

        if frozen_layers:
            self.frozen_params2(frozen_layers)

        if frozenall_skip_layers:
            self.frozen_params(frozenall_skip_layers)

        if reset_head_weight:
            self.reset_head_weight()

        assert frozen_head_levels is None or isinstance(frozen_head_levels, tuple)
        if frozen_head_levels:
            self.to_frozen_head_weights_by_levels(frozen_head_levels)

        # self.load_head_weight('./train_result/scrfd_500m_bnkps_v2/epoch_990.pth')


        ###################### init kd teacher model ######################
        self.use_kd_loss = False
        if kd_teacher_model:
            self.kd_teacher_model = build_detector(kd_teacher_model)
            print('kd_teacher_model: ', self.kd_teacher_model)

            _ckpt = 'weights/scrfd_10g_bnkps.pth'
            _stat_dict = torch.load(_ckpt)['state_dict']
            self.kd_teacher_model.load_state_dict(_stat_dict, strict=True)
            for n, m in self.kd_teacher_model.named_parameters():
                m.requires_grad = False
            print('load teacher_model.')

            self.bbox_head.kd_teacher_model = self.kd_teacher_model
            if hasattr(self.bbox_head, 'ext_init_kd_module'):
                self.bbox_head.ext_init_kd_module(56)
            self.use_kd_loss = True



    def load_head_weight(self, pretrained, only_kps=True):
        import os.path as osp
        if osp.exists(pretrained):
            ckpt = torch.load(pretrained)
            from pprint import pprint
            pprint(type(ckpt))

            _st = {k:v for k,v in ckpt['state_dict'].items() if 'stride_kps' in k}
            print(_st.keys())
            self.bbox_head.load_state_dict(_st, False)


    def frozen_params(self, skip_layernames: tuple):

        """
        冻结大部分层，跳过少数层
        """
        assert isinstance(skip_layernames, tuple)
        for n, m in self.named_parameters():
            _ns = [s for s in skip_layernames if n.find(s) >= 0]
            if 0 == len(_ns):
                print(f'frozen layer_params: {n}')
                m.requires_grad = False

    def frozen_params2(self, select_layernames: tuple):

        """
        冻结少部分层
        Args:
        select_layernames:

        Returns:

        """
        assert isinstance(select_layernames, tuple)
        for n, m in self.named_parameters():
            _ns = [s for s in select_layernames if n.find(s) >= 0]
            # print(_ns)
            if 0 < len(_ns):
                print(f'frozen layer_params: {n}')
                m.requires_grad = False

    def reset_head_weight(self):

        """
        重置 head weights
        """
        if self.bbox_head:
            for m in self.bbox_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

            print(f'warning: reset bbox_head weights...')

    def to_frozen_head_weights_by_levels(self, frozen_head_levels=()):
        """
        冻结Head里 指定尺度的分类回归参数
        """
        for i, (cls_m, reg_m) in enumerate(zip(self.bbox_head.cls_convs, self.bbox_head.reg_convs)):
            if i in frozen_head_levels:
                for n, m in cls_m.named_parameters():
                    m.requires_grad = False
                    print(f'frozen bbox.head(level {i}) layer_params: {n}')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)

        if self.use_kd_loss:
            t_x = self.kd_teacher_model.extract_feat(img)
            self.bbox_head.calcu_kd_loss(t_x,x)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypointss, gt_bboxes_ignore)
        return losses



    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # print(len(outs))
        if self.to_onnx_flag or torch.onnx.is_in_onnx_export():
            print('single_stage.py in-onnx-export')
            cls_score, bbox_pred, kps_pred = outs
            for c in cls_score:
                print(c.shape)
            for c in bbox_pred:
                print(c.shape)
            # print(outs[0].shape, outs[1].shape)
            if self.bbox_head.use_kps:
                for c in kps_pred:
                    print(c.shape)
                return (cls_score, bbox_pred, kps_pred)
            else:
                return (cls_score, bbox_pred)
            # return outs
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        # skip post-processing when exporting to ONNX
        # if torch.onnx.is_in_onnx_export():
        #    return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list

            ## TODO: with kps
            # bbox2result_add_kps(det_bboxes, det_labels, det_kps,  self.bbox_head.num_classes)
            # for det_bboxes, det_kps, det_labels in bbox_list
        ]
        return bbox_results

    def feature_test(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
