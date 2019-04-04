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

package org.apache.spark.mllib.recommendation

import breeze.linalg._
import scala.collection.mutable.ListBuffer

import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD


case class productScore(productId : Int,
                        score: Double,
                        percentage: Double)

case class productExplanation(explainedPid: Int,
                        score: Double,
                        productScores: Array[productScore])

case class userExplanation(user: Int,
                           productExplanations: Array[productExplanation])



@SerialVersionUID(100L)
class ALSExplain extends Serializable {
  def explain( prodFactors: RDD[(Int, Array[Double])],
               ratings: RDD[Rating], lambda: Double, alpha: Double):
  RDD[userExplanation] = {
    val sc = prodFactors.context
    val indexedProdFactors = prodFactors.zipWithIndex().cache()
    val productIndexLookup = indexedProdFactors.map( v => ( v._1._1, v._2))
    val productToIndex = indexedProdFactors.map( v => (v._1._1, (v._2, v._1._2)))
    val indexedFactors = indexedProdFactors.map( v => (v._2, v._1._2)).cache()

    val YIndexMatrix = toIndexedMatrix(indexedFactors)
    val Y = YIndexMatrix.toBlockMatrix().cache()
    val YT = Y.transpose.cache()
    val YTY = YT.multiply(Y).cache()
    // printIndexedMatrix(YTY.toIndexedRowMatrix())

    val local_YT = sc.broadcast(YT.toLocalMatrix().asBreeze).value
    val local_Y = sc.broadcast(Y.toLocalMatrix().asBreeze).value
    val local_YTY = sc.broadcast(YTY.toLocalMatrix().asBreeze).value
    val indexCollect = productIndexLookup.collect()

    val lambdaI = sc.broadcast(diag(breeze.linalg.DenseVector(List.fill(local_YTY.rows)
    (lambda).toArray))).value

    val indexMap = sc.broadcast(indexCollect
      .map(keyValue => (keyValue._1, keyValue._2)).toMap).value
    val productMap = sc.broadcast(indexCollect
      .map( keyValue => (keyValue._2, keyValue._1)).toMap).value

    val userExplanationRDD = ratings
      .map( rating => (rating.product, rating))
      .join(productToIndex)
      .map{
        case (_, joinedRow) =>
          (joinedRow._1.user, (joinedRow._2, (joinedRow._1.rating * alpha)))
      }.groupByKey().map( group => userExplanation(group._1,
      userExlain(group._2.toArray, local_YT, local_Y, local_YTY, lambdaI, indexMap, productMap)))
    userExplanationRDD
  }

  def userExlain(array: Array[((Long, Array[Double]), Double)], local_YT: Matrix[Double],
                 local_Y: Matrix[Double],
                 local_YTY: Matrix[Double], lambdaI: Matrix[Double],
                 indexLookup: Map[Int, Long],
                 productLookup: Map[Long, Int]) : Array[productExplanation] = {

    val YTCUMinusI = array.map(row => (row._1._1, row._1._2.map(_ * row._2)))
      .toSeq.sortBy(_._1).map(_._2.toSeq)
    val YTCUMinusIMatrix = new breeze.linalg.
    DenseMatrix(local_YTY.rows, array.length, YTCUMinusI.flatten.toArray)

    val YTCU = array.map(row => (row._1._1, row._1._2.map( _ * (row._2 + 1))))
      .toSeq.sortBy(_._1).map(_._2.toSeq)
    val YTCUMatrix = new breeze.linalg.
    DenseMatrix(local_YTY.rows, array.length, YTCU.flatten.toArray)

    val cuIndex = array.sortBy(_._1._1).map(_._1._1)
      .zipWithIndex.map( row => (row._2, row._1)).toMap

    val Y = array.map(row => (row._1._1, row._1._2)).toSeq.sortBy(_._1).map(_._2.toSeq)
    val YMatrix = new breeze.linalg.DenseMatrix(local_YTY.rows, array.length, Y.flatten.toArray).t

    val A = YTCUMinusIMatrix * YMatrix
    val W = inv(local_YTY.toDenseMatrix + A.toDenseMatrix + lambdaI.toDenseMatrix).toDenseMatrix
    val S = (local_Y * W * YTCUMatrix).toDenseMatrix
    S(*, ::).map( x => generateExplain(x, cuIndex, productLookup))
      .toArray.zipWithIndex.map(
      row => productExplanation(productLookup(row._2), row._1._1, row._1._2))
      .sortBy(_.explainedPid)
  }

  def generateExplain(row: breeze.linalg.Vector[Double],
                      cuIndex: Map[Int, Long],
                      productLookup: Map[Long, Int]): (Double, Array[productScore]) =
  {
    val result : ListBuffer[productScore] = new ListBuffer[productScore]
    var sum: Double = 0.0
    for ( key <- cuIndex ) {
      val score = row(key._1)
      sum = sum + score
    }

    for ( key <- cuIndex ) {
      val score = row(key._1)
      val item = new productScore(
        productId = productLookup(key._2),
        score = score,
        percentage = score/sum
      )
      result += item
    }

    (sum, result.sortBy(-_.percentage).take(2).toArray)
  }

  def toIndexedMatrix(features: RDD[(Long, Array[Double])]): IndexedRowMatrix = {
    val rows = features.map { case (i, xs) => IndexedRow(i, Vectors.dense(xs)) }
    new IndexedRowMatrix(rows)
  }

  def printIndexedMatrix(matrix: IndexedRowMatrix): Unit = {
    matrix.rows.map( v => (v.index, v.vector.toArray.mkString(","))).collect().foreach(println)
  }

}
