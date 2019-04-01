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
                        cu: Double,
                        similarity: Double,
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

    val userRows = ratings.map( rating => (rating.product, rating))
      .join(productIndexLookup)
      .map( joined => ( joined._2._1.user, (joined._2._2, joined._2._1.rating * alpha)))
      .groupByKey().repartition(1)
      .map( group => userExplanation( group._1,
        process(group._2.toArray, local_YT,
          local_Y, local_YTY, indexMap, productMap, lambdaI ))).cache()
    userRows
    // userRows.saveAsTextFile("output")

  }

  def process(array: Array[(Long, Double)], YT: Matrix[Double],
              Y: Matrix[Double], local_YTY: Matrix[Double], indexLookup: Map[Int, Long],
              productLookup: Map[Long, Int], lambdaI: Matrix[Double])
  : Array[productExplanation] = {
  // Array[(Int, (Double, List[Map[String, Double]]))] = {
    var list : List[Double] = List.fill(Y.rows)(0.0)
    for ( key <- array ) {
      list = list.updated(key._1.toInt, key._2)
    }
    val CU = diag(breeze.linalg.DenseVector(list.toArray))

    val A = YT.toDenseMatrix * CU.toDenseMatrix * Y.toDenseMatrix
    val W = inv(local_YTY.toDenseMatrix + A.toDenseMatrix + lambdaI.toDenseMatrix)
    val S = (Y * W * YT).toDenseMatrix
    val result = S(IndexedSeq(
      indexLookup(1).toInt, indexLookup(2).toInt), ::)
    S(*, ::).map( x => generateExplain(x, array, productLookup))
      .toArray.zipWithIndex.map(
      row => productExplanation(productLookup(row._2.toLong), row._1._1, row._1._2))
  }

  def generateExplain(row: breeze.linalg.Vector[Double],
                      cu: Array[(Long, Double)],
                      productLookup: Map[Long, Int]): (Double, Array[productScore]) =
  {
    val result : ListBuffer[productScore] = new ListBuffer[productScore]
    var sum: Double = 0.0
    for ( key <- cu ) {
      val score = row(key._1.toInt) * (1 + key._2)
      sum = sum + score
    }

    for ( key <- cu ) {
      val score = row(key._1.toInt) * (1 + key._2)
      val item = new productScore(
        productId = productLookup(key._1),
        cu = (1 + key._2),
        similarity = row(key._1.toInt),
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
