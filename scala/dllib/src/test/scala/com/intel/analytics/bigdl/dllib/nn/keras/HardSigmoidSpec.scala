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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.dllib.nn.HardSigmoid


class HardSigmoidSpec extends KerasBaseSpec {

  "Hard_sigmoid" should "be ok" in {
    val sigmoidCode =
      """
        |input_tensor = Input(shape=[5])
        |input = np.random.uniform(0, 1, [4, 5])
        |output_tensor = Activation(activation="hard_sigmoid")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val hardSigmoid = HardSigmoid[Float]()

    checkOutputAndGrad(hardSigmoid, sigmoidCode)

  }
}
