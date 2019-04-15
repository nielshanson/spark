/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.recommendation

import scala.collection.mutable.WrappedArray

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

@SerialVersionUID(100L)
class ALSExplain extends Serializable{
  def explain(itemDF: DataFrame,
              dataset: Dataset[_],
              userCol: String,
              itemCol: String,
              ratingCol: String,
              topExplanation: Int,
              regParam: Double = 0.1,
              alpha: Double = 1.0): DataFrame = {

    val spark = itemDF.sparkSession
    val ratings = dataset
      .select(userCol, itemCol, ratingCol)
      .rdd
      .map { row =>
        Rating(row.getLong(0).toInt, row.getLong(1).toInt, row.getDouble(2))
      }

    val prodFactors: RDD[(Int, Array[Double])] =
      itemDF.rdd.map(row => (row.getInt(0),
        row.getAs[WrappedArray[Float]](1).toArray.map(_.toDouble)))
    val explanation = new org.apache.spark.mllib.recommendation.ALSExplain()
      .explain(prodFactors, ratings, regParam, alpha, topExplanation)
    val df = spark.createDataFrame(explanation)
    df
  }
}

