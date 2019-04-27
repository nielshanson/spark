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

import org.apache.spark.ml.recommendation.ALS._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row, SparkSession}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

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
    import dataset.sparkSession.implicits._
    // validateAndTransformSchema((StructType)dataset.schema, userCol, itemCol, ratingCol)

    val r = if (col(ratingCol) != "") col(ratingCol).cast(FloatType) else lit(1.0f)
    val ratings = dataset
      .select(checkedCast(col(userCol)), checkedCast(col(itemCol)), r)
      .rdd
      .map { row =>
        Rating(row.getInt(0), row.getInt(1), row.getFloat(2))
      }

    val prodFactors: RDD[(Int, Array[Double])] =
      itemDF.rdd.map(row => (row.getInt(0),
        row.getAs[WrappedArray[Float]](1).toArray.map(_.toDouble)))
    val explanation = new org.apache.spark.mllib.recommendation.ALSExplain()
      .explain(prodFactors, ratings, regParam, alpha, topExplanation)
    val df = dataset.sparkSession.createDataFrame(explanation)
    df
  }

  /**
   * Attempts to safely cast a user/item id to an Int. Throws an exception if the value is
   * out of integer range or contains a fractional part.
   */
  protected[recommendation] val checkedCast = udf { (n: Any) =>
    n match {
      case v: Int => v // Avoid unnecessary casting
      case v: Number =>
        val intV = v.intValue
        // Checks if number within Int range and has no fractional part.
        if (v.doubleValue == intV) {
          intV
        } else {
          throw new IllegalArgumentException(s"ALS only supports values in Integer range " +
            s"and without fractional part for columns col{col(userCol)} and col{col(itemCol)}. " +
            s"Value coln was either out of Integer range or contained a fractional part that " +
            s"could not be converted.")
        }
      case _ => throw new IllegalArgumentException(s"ALS only supports values in Integer range " +
        s"for columns col{col(userCol)} and col{col(itemCol)}. Value coln was not numeric.")
    }

  def explainUser(userDF: DataFrame,
                  itemDF: DataFrame,
                  dataset: Dataset[_],
                  userCol: String,
                  itemCol: String,
                  ratingCol: String,
                  topExplanation: Int,
                  regParam: Double = 0.1,
                  alpha: Double = 1.0): DataFrame = {

    val spark = itemDF.sparkSession
    val ratings = dataset
        .join(userDF, userCol)
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

