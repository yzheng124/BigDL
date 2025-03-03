/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.dllib.feature.image

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import org.apache.logging.log4j.LogManager
import org.opencv.imgcodecs.Imgcodecs

/**
 * Transform byte array(original image file in byte) to OpenCVMat
 * @param byteKey key that maps byte array
 * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
 *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
 */
class ImageBytesToMat(byteKey: String = ImageFeature.bytes,
                      imageCodec: Int = Imgcodecs.IMREAD_UNCHANGED) extends ImageProcessing {

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform(_))
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    ImageBytesToMat.transform(feature, byteKey, imageCodec)
  }
}

object ImageBytesToMat {
  val logger = LogManager.getLogger(getClass)

  def apply(byteKey: String = ImageFeature.bytes,
            imageCodec: Int = Imgcodecs.IMREAD_UNCHANGED): ImageBytesToMat =
    new ImageBytesToMat(byteKey, imageCodec)

  def transform(feature: ImageFeature, byteKey: String, imageCodec: Int): ImageFeature = {
    if (!feature.isValid) return feature
    val bytes = feature[Array[Byte]](byteKey)
    var mat: OpenCVMat = null
    try {
      Log4Error.invalidInputError(null != bytes && bytes.length > 0,
        "image file bytes should not be empty")
      mat = OpenCVMethod.fromImageBytes(bytes, imageCodec)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        val uri = feature.uri()
        logger.info(s"The convertion from bytes to mat fails for $uri")
        feature(ImageFeature.originalSize) = (-1, -1, -1)
        feature.isValid = false
    }
    feature
  }
}

